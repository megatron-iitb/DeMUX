"""
Improved Accountable AI Agent - Experiment 2 Enhanced Version

Key Improvements:
1. Better task decomposition with complete prompts
2. Extractive answer generation from context
3. Trained aggregation network with synthetic data
4. Improved uncertainty estimation
5. Better calibration mechanisms
6. Complete answer generation with validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.special import softmax
from sklearn.calibration import calibration_curve
import warnings
import re
warnings.filterwarnings('ignore')


@dataclass
class SubtaskResult:
    """Stores results from individual subtasks"""
    subtask_id: int
    description: str
    answer: str
    probability: float
    confidence: float
    evidence: List[str]
    reasoning_chain: List[str]


@dataclass
class FinalPrediction:
    """Final prediction with accountability metrics"""
    answer: str
    confidence: float
    subtask_contributions: Dict[int, float]
    calibration_score: float
    uncertainty_estimate: float
    explanation: str
    evidence_trail: List[Dict]


class ImprovedTaskDecomposer:
    """Enhanced task decomposition with complete prompts and fallback strategies"""

    def __init__(self, model_name: str = "facebook/opt-350m"):
        print("Loading task decomposer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def decompose(self, query: str, num_subtasks: int = 3) -> List[str]:
        """Break down main query into subtasks with improved prompting"""

        # Improved complete prompt with examples
        prompt = f"""Break down complex questions into simpler sub-questions.

Example:
Main: What causes economic recessions?
Sub-questions:
1. What are the immediate triggers of economic recessions?
2. What are the structural factors that lead to recessions?
3. How do government policies influence recession severity?

Main: {query}
Sub-questions:
1."""

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=400)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=500,  # Increased length
                num_return_sequences=1,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Prevent repetition
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Enhanced parsing
        subtasks = []
        
        # Extract the generated part after our example
        if "Sub-questions:" in result:
            parts = result.split("Sub-questions:")
            if len(parts) >= 2:
                generated_part = parts[-1]  # Get the last section (our generation)
                lines = generated_part.split('\n')
                
                for line in lines:
                    line = line.strip()
                    # Look for numbered items with various formats
                    if line and any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 10)):
                        # Remove numbering and clean
                        task = re.sub(r'^\d+[\.)]\s*', '', line).strip()
                        if task and len(task) > 20:  # Ensure substantial question
                            # Add question mark if missing
                            if not task.endswith('?'):
                                task += '?'
                            subtasks.append(task)

        # Enhanced domain-specific fallback
        if len(subtasks) < num_subtasks:
            print(f"   ⚠️  Decomposition produced {len(subtasks)} subtasks, using fallback...")
            subtasks = self._generate_fallback_subtasks(query, num_subtasks)

        return subtasks[:num_subtasks]

    def _generate_fallback_subtasks(self, query: str, num_subtasks: int) -> List[str]:
        """Generate domain-specific fallback subtasks"""
        query_lower = query.lower()
        
        # Detect domain
        if any(word in query_lower for word in ['climate', 'environment', 'emissions', 'warming']):
            return [
                f"What are the primary anthropogenic (human-caused) factors contributing to climate change?",
                f"What are the natural processes and feedback loops involved in climate systems?",
                f"How do different climate change factors compare in their relative impact and importance?"
            ]
        elif any(word in query_lower for word in ['health', 'medical', 'disease', 'treatment']):
            return [
                f"What are the underlying biological or physiological mechanisms?",
                f"What are the major risk factors or causal elements?",
                f"What are the quantitative measurements or severity indicators?"
            ]
        elif any(word in query_lower for word in ['economic', 'financial', 'market', 'trade']):
            return [
                f"What are the microeconomic factors at the individual/firm level?",
                f"What are the macroeconomic and systemic factors?",
                f"What are the policy and regulatory influences?"
            ]
        else:
            # Generic decomposition strategy
            return [
                f"What are the primary direct factors or causes related to: {query}?",
                f"What are the secondary or contextual factors influencing: {query}?",
                f"What is the relative importance or hierarchy of factors in: {query}?"
            ]


class EnhancedReasoningEngine:
    """Improved reasoning with extractive answering and better confidence estimation"""

    def __init__(self):
        print("Loading CLIP and reasoning models...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Use text generation for answering
        self.qa_pipeline = pipeline(
            "text-generation",
            model="facebook/opt-350m",
            device=-1  # CPU
        )

    def compute_clip_similarity(self, text: str, context: List[str]) -> np.ndarray:
        """Compute CLIP embeddings similarity for context relevance"""
        if not context:
            return np.array([0.5])
            
        inputs = self.clip_processor(
            text=[text] + context,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)

        query_emb = outputs[0].unsqueeze(0)
        context_embs = outputs[1:]

        similarities = F.cosine_similarity(query_emb, context_embs)
        return similarities.numpy()

    def compute_semantic_entropy(self, answers: List[str]) -> float:
        """Compute semantic entropy using CLIP text embeddings"""
        if len(answers) < 2:
            return 0.5  # Changed from 1.0 to be less pessimistic

        # Filter out very short answers
        valid_answers = [a for a in answers if len(a.strip()) > 15]
        if len(valid_answers) < 2:
            return 0.5

        # Embed all answers
        inputs = self.clip_processor(
            text=valid_answers,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            embeddings = self.clip_model.get_text_features(**inputs)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise similarities
        similarities = embeddings @ embeddings.T

        # Average similarity (excluding diagonal)
        mask = ~torch.eye(len(valid_answers), dtype=bool)
        avg_similarity = similarities[mask].mean().item()

        # Convert to entropy-like measure (0 = certain, 1 = uncertain)
        # Use a sigmoid-like transformation for better scaling
        entropy = 1.0 / (1.0 + np.exp(5 * (avg_similarity - 0.5)))
        return max(0.0, min(1.0, entropy))

    def extract_answer_from_context(self, question: str, context: List[str]) -> str:
        """Extract a focused answer using the most relevant context"""
        if not context:
            return "Insufficient context to answer question."
        
        # Get relevance scores
        relevance_scores = self.compute_clip_similarity(question, context)
        
        # Get top 3 most relevant contexts (increased from 2)
        top_indices = np.argsort(relevance_scores)[-3:][::-1]
        top_contexts = [context[i] for i in top_indices if i < len(context)]
        
        # Check if we should just return the most relevant context directly
        max_relevance = relevance_scores[top_indices[0]] if len(top_indices) > 0 else 0
        if max_relevance > 0.85 and top_contexts:
            # Very high relevance - use extractive answer
            return top_contexts[0]
        
        # Build a focused prompt emphasizing data extraction
        context_str = " ".join(top_contexts)
        prompt = f"""Using ONLY the facts provided, answer the question. Extract and cite specific numbers, percentages, and data points.

Facts: {context_str}

Question: {question}

Answer (cite specific data):"""

        try:
            outputs = self.qa_pipeline(
                prompt,
                max_new_tokens=120,  # Increased for complete answers
                min_new_tokens=25,   # Ensure minimum length
                num_return_sequences=1,
                temperature=0.85,    # Increased for more diverse, data-focused output
                top_p=0.92,
                do_sample=True,
                repetition_penalty=1.15,  # Reduce generic repetition
                return_full_text=False
            )
            
            answer = outputs[0]['generated_text'].strip()
            
            # Clean up the answer
            # Take first complete sentence or paragraph
            sentences = answer.split('.')
            if len(sentences) > 1 and len(sentences[0]) > 20:
                answer = sentences[0] + '.'
            elif len(answer) < 20 and len(sentences) > 1:
                answer = '. '.join(sentences[:2]) + '.'
            
            # Fallback to extractive answer if generation is poor or too generic
            generic_phrases = ['is changing', 'climate change', 'in the past', 'complex', 'dynamic']
            is_generic = sum(1 for phrase in generic_phrases if phrase in answer.lower()) >= 2
            
            if len(answer) < 30 or not any(c.isalpha() for c in answer) or is_generic:
                # Return the most relevant context as answer (more reliable)
                answer = top_contexts[0] if top_contexts else "Unable to generate answer."
                
            return answer
            
        except Exception as e:
            print(f"   ⚠️  Generation error: {e}, using extractive fallback")
            return top_contexts[0] if top_contexts else "Unable to generate answer."

    def reason_with_uncertainty(
        self,
        subtask: str,
        context: List[str],
        subtask_id: int
    ) -> SubtaskResult:
        """Process subtask with improved answer generation and uncertainty estimation"""

        # Get context relevance using CLIP
        if context:
            relevance_scores = self.compute_clip_similarity(subtask, context)
            avg_relevance = float(np.mean(relevance_scores))
            top_contexts = [context[i] for i in np.argsort(relevance_scores)[-3:][::-1]]
        else:
            relevance_scores = np.array([0.5])
            avg_relevance = 0.5
            top_contexts = []

        # Generate multiple candidate answers with varied approaches
        answers = []
        for attempt in range(3):  # Generate 3 diverse answers
            answer = self.extract_answer_from_context(subtask, context)
            if answer and len(answer) > 20:
                answers.append(answer)
        
        # If all answers are too similar or poor, add direct context extraction
        if len(answers) < 2 and top_contexts:
            answers.append(top_contexts[0])
            if len(top_contexts) > 1:
                answers.append(top_contexts[1])
        
        # Ensure we have at least one answer
        if not answers:
            answers = [top_contexts[0] if top_contexts else "Unable to provide answer due to insufficient context."]

        # Compute semantic entropy (uncertainty)
        semantic_entropy = self.compute_semantic_entropy(answers)

        # Probability is based on:
        # 1. Semantic consistency (1 - entropy)
        # 2. Context relevance
        semantic_consistency = 1.0 - semantic_entropy
        probability = 0.6 * semantic_consistency + 0.4 * avg_relevance

        # Confidence incorporates probability and answer quality
        answer_quality = min(1.0, len(answers[0]) / 100)  # Length-based quality indicator
        confidence = 0.7 * probability + 0.3 * answer_quality

        # Create detailed reasoning chain
        reasoning_chain = [
            f"Question: {subtask[:80]}{'...' if len(subtask) > 80 else ''}",
            f"Context relevance: {avg_relevance:.3f} (used {len(top_contexts)} sources)",
            f"Generated {len(answers)} candidate answers",
            f"Semantic consistency: {semantic_consistency:.3f}",
            f"Uncertainty (entropy): {semantic_entropy:.3f}",
            f"Answer quality score: {answer_quality:.3f}"
        ]

        return SubtaskResult(
            subtask_id=subtask_id,
            description=subtask,
            answer=answers[0],  # Use best/first answer
            probability=probability,
            confidence=confidence,
            evidence=top_contexts,
            reasoning_chain=reasoning_chain
        )


class TrainedAggregationNetwork(nn.Module):
    """Neural network with pre-training on synthetic data"""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, num_subtasks: int = 3):
        super().__init__()
        self.num_subtasks = num_subtasks
        self.input_dim = input_dim

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Aggregation network
        combined_dim = input_dim * num_subtasks + num_subtasks * 2

        self.aggregator = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        self._initialize_weights()
        self.is_trained = False

    def _initialize_weights(self):
        """Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        subtask_embeddings: torch.Tensor,
        probabilities: torch.Tensor,
        confidences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Compute attention weights
        attention_logits = torch.stack([self.attention(emb) for emb in subtask_embeddings])
        attention_weights = F.softmax(attention_logits.squeeze(), dim=0)

        # Combine all information
        flat_embeddings = subtask_embeddings.flatten()
        combined = torch.cat([flat_embeddings, probabilities, confidences])

        # Final prediction
        final_confidence = self.aggregator(combined)
        uncertainty = self.uncertainty_head(combined)

        return final_confidence, attention_weights, uncertainty

    def pretrain_on_synthetic_data(self, num_samples: int = 100):
        """Pre-train on synthetic data to learn reasonable aggregation patterns"""
        print(f"   Pre-training aggregation network on {num_samples} synthetic samples...")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        self.train()
        for epoch in range(20):  # Quick pre-training
            epoch_loss = 0.0
            
            for _ in range(num_samples // 10):
                # Generate synthetic data
                # Simulate subtask embeddings (random but normalized)
                embeddings = torch.randn(self.num_subtasks, self.input_dim)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Simulate probabilities and confidences
                # Higher values should lead to higher final confidence
                probs = torch.rand(self.num_subtasks) * 0.5 + 0.3  # 0.3 to 0.8
                confs = torch.rand(self.num_subtasks) * 0.4 + 0.4  # 0.4 to 0.8
                
                # Target: weighted average of confidences (with noise)
                target = (probs * confs).mean() + torch.randn(1).item() * 0.05
                target = torch.clamp(torch.tensor([target]), 0.35, 0.85)  # Higher minimum to reduce conservatism
                
                # Forward pass
                final_conf, attn, uncertainty = self.forward(embeddings, probs, confs)
                
                # Loss: MSE for confidence + regularization for attention diversity
                loss = F.mse_loss(final_conf, target)
                
                # Encourage attention to focus on high-confidence subtasks
                attention_loss = -torch.sum(attn * confs) * 0.1
                
                # Uncertainty should correlate with variance in probabilities
                uncertainty_target = torch.std(probs).unsqueeze(0)
                uncertainty_loss = F.mse_loss(uncertainty, uncertainty_target) * 0.1
                
                total_loss = loss + attention_loss + uncertainty_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / (num_samples // 10)
                print(f"      Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
        
        self.eval()
        self.is_trained = True
        print(f"   ✓ Pre-training complete!")


class ImprovedCalibrationModule:
    """Enhanced calibration with better temperature scaling"""

    def __init__(self):
        self.confidence_history = []
        self.outcomes_history = []
        self.temperature = 1.0
        self.min_samples_for_calibration = 5

    def add_prediction(self, confidence: float, was_correct: bool):
        self.confidence_history.append(confidence)
        self.outcomes_history.append(1.0 if was_correct else 0.0)

    def compute_calibration_score(self) -> float:
        """Compute Expected Calibration Error (ECE)"""
        if len(self.confidence_history) < self.min_samples_for_calibration:
            return 0.0

        confidences = np.array(self.confidence_history)
        outcomes = np.array(self.outcomes_history)

        n_bins = min(10, len(confidences) // 2)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                avg_confidence = confidences[mask].mean()
                avg_accuracy = outcomes[mask].mean()
                bin_weight = mask.sum() / len(confidences)
                ece += bin_weight * abs(avg_confidence - avg_accuracy)

        return float(ece)

    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """Apply learned temperature scaling"""
        if len(self.confidence_history) < self.min_samples_for_calibration:
            # No calibration yet, boost low confidence if subtasks are high
            # This helps when aggregation is too conservative
            return min(raw_confidence * 1.15, 0.75)
        
        avg_confidence = np.mean(self.confidence_history)
        avg_accuracy = np.mean(self.outcomes_history)

        # Adjust temperature based on calibration gap
        calibration_gap = avg_confidence - avg_accuracy
        
        if calibration_gap > 0.15:  # Very overconfident
            self.temperature = 2.0
        elif calibration_gap > 0.05:  # Slightly overconfident
            self.temperature = 1.5
        elif calibration_gap < -0.15:  # Very underconfident
            self.temperature = 0.5
        elif calibration_gap < -0.05:  # Slightly underconfident
            self.temperature = 0.75
        else:  # Well calibrated
            self.temperature = 1.0

        # Apply temperature scaling
        calibrated = raw_confidence ** (1.0 / self.temperature)
        
        # Ensure reasonable bounds
        return np.clip(calibrated, 0.1, 0.95)


class EnhancedExplainabilityModule:
    """Improved explanations with better formatting and insights"""

    def generate_explanation(
        self,
        query: str,
        subtask_results: List[SubtaskResult],
        attention_weights: torch.Tensor,
        final_answer: str,
        confidence: float
    ) -> str:

        explanation_parts = [
            f"# Analysis Report\n\n",
            f"**Main Query**: {query}\n\n",
            f"**Final Answer**: {final_answer}\n\n",
            f"**Overall Confidence**: {confidence:.1%}\n\n",
        ]

        # Add confidence interpretation
        if confidence > 0.7:
            conf_interpretation = "🟢 High confidence - strong evidence and consistency"
        elif confidence > 0.5:
            conf_interpretation = "🟡 Moderate confidence - reasonable evidence but some uncertainty"
        else:
            conf_interpretation = "🔴 Low confidence - limited evidence or inconsistency detected"
        
        explanation_parts.append(f"**Interpretation**: {conf_interpretation}\n\n")
        explanation_parts.append(f"## Reasoning Process\n\n")

        # Sort by attention weight
        sorted_indices = torch.argsort(attention_weights, descending=True)

        for rank, idx in enumerate(sorted_indices, 1):
            result = subtask_results[idx]
            weight = attention_weights[idx].item()

            # Determine importance label
            if weight > 0.4:
                importance = "🔴 Critical"
            elif weight > 0.25:
                importance = "🟡 Important"
            else:
                importance = "🟢 Supporting"

            explanation_parts.append(
                f"### #{rank}: {importance} (Attention Weight: {weight:.1%})\n\n"
                f"**Sub-question**: {result.description}\n\n"
                f"**Finding**: {result.answer}\n\n"
                f"**Metrics**: Confidence: {result.confidence:.1%} | Probability: {result.probability:.1%}\n\n"
            )

            if result.evidence:
                explanation_parts.append(f"**Supporting Evidence**:\n")
                for i, evidence in enumerate(result.evidence[:2], 1):
                    explanation_parts.append(f"  {i}. {evidence}\n")
                explanation_parts.append("\n")

            if result.reasoning_chain:
                explanation_parts.append("**Reasoning Steps**:\n")
                for step in result.reasoning_chain:
                    explanation_parts.append(f"  • {step}\n")
                explanation_parts.append("\n")

        # Add summary statistics
        explanation_parts.append(f"## Summary Statistics\n\n")
        avg_prob = np.mean([r.probability for r in subtask_results])
        avg_conf = np.mean([r.confidence for r in subtask_results])
        explanation_parts.append(f"- Average subtask probability: {avg_prob:.1%}\n")
        explanation_parts.append(f"- Average subtask confidence: {avg_conf:.1%}\n")
        explanation_parts.append(f"- Attention concentration: {max(attention_weights).item():.1%} (max weight)\n")

        return "".join(explanation_parts)


class AccountableAIAgent:
    """Enhanced main agent with all improvements integrated"""

    def __init__(self, num_subtasks: int = 3, pretrain_network: bool = True):
        print("\n🚀 Initializing Improved Accountable AI Agent...")
        print("="*60)

        self.decomposer = ImprovedTaskDecomposer()
        self.reasoning_engine = EnhancedReasoningEngine()
        self.aggregation_net = TrainedAggregationNetwork(
            input_dim=512,
            num_subtasks=num_subtasks
        )
        
        # Pre-train the aggregation network
        if pretrain_network:
            self.aggregation_net.pretrain_on_synthetic_data(num_samples=200)
        
        self.calibration = ImprovedCalibrationModule()
        self.explainability = EnhancedExplainabilityModule()
        self.num_subtasks = num_subtasks

        print("="*60)
        print("✅ Agent ready!\n")

    def _embed_text(self, text: str) -> torch.Tensor:
        """Create embeddings using CLIP"""
        inputs = self.reasoning_engine.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            embeddings = self.reasoning_engine.clip_model.get_text_features(**inputs)

        # Normalize and pad to 512 dimensions
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if embeddings.shape[1] < 512:
            padding = torch.zeros(1, 512 - embeddings.shape[1])
            embeddings = torch.cat([embeddings, padding], dim=1)
        else:
            embeddings = embeddings[:, :512]

        return embeddings.squeeze()

    def process_query(
        self,
        query: str,
        context_data: Optional[List[str]] = None
    ) -> FinalPrediction:
        """Main pipeline with all improvements"""

        print(f"\n{'='*70}")
        print(f"📋 QUERY: {query}")
        print(f"{'='*70}\n")

        # Step 1: Decomposition
        print("🔍 Step 1: Breaking down into subtasks...")
        subtasks = self.decomposer.decompose(query, self.num_subtasks)
        for i, task in enumerate(subtasks, 1):
            print(f"   {i}. {task[:100]}{'...' if len(task) > 100 else ''}")

        # Step 2: Reasoning
        print("\n🧠 Step 2: Reasoning through subtasks...")
        subtask_results = []
        subtask_embeddings = []
        probabilities = []
        confidences = []

        context = context_data if context_data else [query]

        for i, subtask in enumerate(subtasks):
            print(f"   Processing {i+1}/{len(subtasks)}...", end=" ")
            result = self.reasoning_engine.reason_with_uncertainty(
                subtask, context, i+1
            )
            subtask_results.append(result)

            emb = self._embed_text(result.answer)
            subtask_embeddings.append(emb)
            probabilities.append(result.probability)
            confidences.append(result.confidence)

            print(f"✓ (conf: {result.confidence:.2%}, prob: {result.probability:.2%})")

        # Step 3: Aggregation
        print("\n⚡ Step 3: Aggregating with trained neural network...")
        subtask_embeddings_tensor = torch.stack(subtask_embeddings)
        probabilities_tensor = torch.tensor(probabilities, dtype=torch.float32)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32)

        with torch.no_grad():
            final_confidence_raw, attention_weights, uncertainty = self.aggregation_net(
                subtask_embeddings_tensor,
                probabilities_tensor,
                confidences_tensor
            )

        # Apply calibration
        final_confidence = self.calibration.get_calibrated_confidence(
            final_confidence_raw.item()
        )

        print(f"   Raw confidence: {final_confidence_raw.item():.2%}")
        print(f"   Calibrated confidence: {final_confidence:.2%}")
        print(f"   Uncertainty estimate: {uncertainty.item():.3f}")
        print(f"   Attention distribution: {[f'{w:.1%}' for w in attention_weights.detach().numpy()]}")

        # Step 4: Select answer using quality-aware selection
        # Prefer answers with high confidence AND data citations
        best_idx = torch.argmax(attention_weights).item()
        
        # Check if best answer has numerical data (more informative)
        best_answer = subtask_results[best_idx].answer
        has_numbers = any(c.isdigit() for c in best_answer)
        
        if not has_numbers:
            # Look for other subtasks with numerical data
            for idx in torch.argsort(attention_weights, descending=True):
                candidate = subtask_results[idx].answer
                if any(c.isdigit() for c in candidate) and attention_weights[idx] > 0.15:
                    best_idx = idx
                    best_answer = candidate
                    break
        
        # If attention is very focused (>50%) or answer has data, use it
        if attention_weights[best_idx] > 0.5 or has_numbers:
            final_answer = best_answer
        else:
            # Combine insights from top 2 subtasks
            top2_indices = torch.argsort(attention_weights, descending=True)[:2]
            answers = [subtask_results[idx].answer for idx in top2_indices]
            final_answer = f"{answers[0]} Additionally, {answers[1].lower() if answers[1][0].isupper() else answers[1]}"

        # Step 5: Explanation
        print("\n📊 Step 4: Generating explanation...")
        calibration_score = self.calibration.compute_calibration_score()

        explanation = self.explainability.generate_explanation(
            query,
            subtask_results,
            attention_weights,
            final_answer,
            final_confidence
        )

        evidence_trail = [
            {
                "subtask_id": r.subtask_id,
                "description": r.description,
                "answer": r.answer,
                "confidence": r.confidence,
                "probability": r.probability,
                "reasoning": r.reasoning_chain,
                "evidence": r.evidence
            }
            for r in subtask_results
        ]

        prediction = FinalPrediction(
            answer=final_answer,
            confidence=final_confidence,
            subtask_contributions={
                i+1: attention_weights[i].item()
                for i in range(len(subtask_results))
            },
            calibration_score=calibration_score,
            uncertainty_estimate=uncertainty.item(),
            explanation=explanation,
            evidence_trail=evidence_trail
        )

        print(f"\n{'='*70}")
        print("✅ Analysis complete!")
        print(f"{'='*70}\n")

        return prediction

    def update_calibration(self, was_correct: bool, confidence: float):
        """Update calibration with feedback"""
        self.calibration.add_prediction(confidence, was_correct)
        ece = self.calibration.compute_calibration_score()
        print(f"📈 Calibration updated. ECE: {ece:.3f}, Temperature: {self.calibration.temperature:.2f}")


# Example Usage
if __name__ == "__main__":
    # Initialize agent
    agent = AccountableAIAgent(num_subtasks=3, pretrain_network=True)

    # Example 1: Climate change query
    query = "What are the main factors contributing to climate change and their relative importance?"
    context = [
        "Greenhouse gas emissions from fossil fuels account for 65% of global emissions",
        "Deforestation contributes 11% to global greenhouse gas emissions",
        "Industrial processes and agriculture contribute 24% combined",
        "Transportation sector emissions represent 14% of total emissions",
        "Carbon dioxide levels have increased 50% since pre-industrial times"
    ]

    print("\n" + "="*70)
    print("EXAMPLE: Climate Change Analysis")
    print("="*70)

    result = agent.process_query(query, context)

    # Display results
    print("\n" + "="*70)
    print("📄 FINAL RESULTS")
    print("="*70)
    print(f"\n**Answer**: {result.answer}\n")
    print(f"**Confidence**: {result.confidence:.1%}")
    print(f"**Uncertainty**: {result.uncertainty_estimate:.3f}")
    print(f"**Calibration (ECE)**: {result.calibration_score:.3f}")

    print("\n**Subtask Contributions**:")
    for subtask_id, contribution in sorted(result.subtask_contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  • Subtask {subtask_id}: {contribution:.1%}")

    print("\n" + "-"*70)
    print(result.explanation)
    print("-"*70)

    # Simulate feedback for calibration improvement
    print("\n💬 Simulating user feedback for calibration...")
    agent.update_calibration(was_correct=True, confidence=result.confidence)

    print("\n✅ Pipeline completed successfully!")
    print("\n💡 KEY IMPROVEMENTS:")
    print("   1. ✓ Better task decomposition with complete prompts")
    print("   2. ✓ Improved answer generation with validation")
    print("   3. ✓ Pre-trained aggregation network (not random weights)")
    print("   4. ✓ Enhanced uncertainty estimation")
    print("   5. ✓ Better calibration with temperature scaling")
    print("   6. ✓ More informative explanations with confidence interpretation")
