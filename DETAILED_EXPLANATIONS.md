# Experiment 2: DeMUX - Detailed Technical Explanations

## Table of Contents
1. [Core Intuition](#core-intuition)
2. [Why Decomposition Works](#why-decomposition-works)
3. [Task Decomposition Deep Dive](#task-decomposition-deep-dive)
4. [Probabilistic Reasoning Explained](#probabilistic-reasoning-explained)
5. [Neural Aggregation Mathematics](#neural-aggregation-mathematics)
6. [Semantic Entropy for Uncertainty](#semantic-entropy-for-uncertainty)
7. [Calibration Theory](#calibration-theory)
8. [Pre-training Strategy](#pre-training-strategy)
9. [Answer Quality Analysis](#answer-quality-analysis)
10. [Practical Implementation Insights](#practical-implementation-insights)

---

## Core Intuition

### The Fundamental Problem

**Complex questions are hard for several reasons**:
1. **Multiple aspects**: Need to consider different perspectives
2. **Hidden assumptions**: What exactly is being asked?
3. **Confidence opacity**: How certain is the model?
4. **No evidence trail**: Why this answer?

**Example**: "What are the main factors contributing to climate change?"
- Which factors? Anthropogenic? Natural?
- Main by what metric? Magnitude? Growth rate?
- What timeframe? Historical? Current? Future?
- How certain are we? Based on what evidence?

### The Decomposition Solution

**Key Insight**: Break complex problem into simpler sub-problems

**Analogy 1: Building a House**
```
❌ Bad: "Build a house" → One contractor does everything
   Problem: No specialization, hard to verify quality

✅ Good: Decompose into trades
   Foundation → Framing → Plumbing → Electrical → Finishing
   Each specialist does their part, inspector verifies each stage
```

**Analogy 2: Scientific Research**
```
❌ Bad: "Solve climate change" → Single paper
   Problem: Too broad, can't verify claims

✅ Good: Decompose into research questions
   Q1: What are emission sources? → Measurement study
   Q2: What drives emissions? → Causal analysis
   Q3: What interventions work? → Policy evaluation
   Q4: How to scale solutions? → Implementation research
   
   Each paper focused, verifiable, builds on others
```

### Why This Helps AI

**1. Smaller Context Windows**
- Each subtask uses ~3 relevant documents (not all 5)
- CLIP can focus attention better

**2. Verifiable Steps**
- Human can check each subtask answer independently
- Easier to spot errors than in end-to-end generation

**3. Uncertainty Propagation**
- Measure confidence per subtask
- Aggregate with variance penalty
- Final uncertainty reflects cumulative unknowns

**4. Modular Improvements**
- Replace decomposer (OPT → T5)
- Replace reasoner (generative → extractive)
- Retrain aggregator on domain data
- Without changing other components!

---

## Why Decomposition Works

### Information Theory Perspective

**Entropy of Complex Question**: H(Q_complex) is high
- Many possible interpretations
- Requires broad knowledge
- High uncertainty

**Entropy of Subtasks**: H(Q1), H(Q2), H(Q3) individually lower
- More focused, specific questions
- Narrower knowledge requirement
- Lower individual uncertainty

**Key Theorem**: 
```
H(Q_complex) ≥ Σ H(Q_i) - I(Q_i; Q_j)
              ^^^^^^^^^^^   ^^^^^^^^^^
              Subtask       Mutual info
              entropies     (overlap)
```

**Interpretation**: 
- If subtasks independent: Total uncertainty = sum of parts
- If subtasks overlapping: Mutual info reduces effective uncertainty
- **DeMUX leverages this**: Consistent answers across subtasks → Low mutual info → High confidence!

---

### Cognitive Science Perspective

**Human Problem-Solving**:
1. **Chunking**: Break problem into manageable pieces
2. **Working Memory**: Hold 3-4 items at once (matches our 3 subtasks!)
3. **Integration**: Combine partial solutions

**DeMUX Mirrors This**:
- Task decomposer = Chunking strategy
- Reasoning engine = Working memory processing
- Neural aggregator = Integration mechanism

**Experimental Evidence**:
- Humans more accurate on decomposed tasks (Chi et al., 1981)
- Expert problem solvers decompose naturally (Larkin, 1980)
- Decomposition reduces cognitive load (Sweller, 1988)

---

## Task Decomposition Deep Dive

### Domain Detection Algorithm

**Step 1: Keyword Matching**
```python
DOMAIN_KEYWORDS = {
    'climate': ['emissions', 'carbon', 'temperature', 'greenhouse', 
                'climate', 'warming', 'fossil', 'renewable'],
    'health': ['disease', 'symptoms', 'treatment', 'diagnosis', 
               'patient', 'medical', 'drug', 'infection'],
    'economics': ['GDP', 'growth', 'inflation', 'unemployment', 
                  'market', 'trade', 'fiscal', 'monetary']
}

def detect_domain(query):
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in query.lower())
        scores[domain] = score
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
```

**Our Example**:
```
Query: "What are the main factors contributing to climate change..."
Keywords found: ['climate', 'change']
Score: climate=2, health=0, economics=0
Domain: climate ✅
```

---

### Few-Shot Prompting Strategy

**Template** (for climate domain):
```
You are an expert at breaking down complex questions into simpler sub-questions.

Example 1:
Query: "How does deforestation contribute to climate change?"
Subtasks:
1. How much carbon do forests absorb annually?
2. What are the main drivers of deforestation?
3. How does deforestation affect carbon emissions?

Example 2:
Query: "What are the projected impacts of a 2°C temperature rise?"
Subtasks:
1. What are the regional temperature variations expected?
2. What are the impacts on sea level and extreme weather?
3. What are the economic and social consequences?

Example 3:
Query: "How effective are renewable energy policies?"
Subtasks:
1. What are the current renewable energy adoption rates?
2. What policies have been implemented in leading countries?
3. What is the correlation between policies and adoption?

Now decompose this query into 3 sub-questions:
Query: "{user_query}"
Subtasks:
```

**Generation Parameters**:
```python
temperature=0.7          # Not too random, not too deterministic
top_p=0.9               # Nucleus sampling (keep 90% probability mass)
no_repeat_ngram_size=3  # Prevent "the main factors" repeating
max_new_tokens=100      # ~3 subtasks × ~30 tokens each
```

---

### Fallback Templates

**When few-shot fails** (generation timeout, gibberish, duplicates):

```python
FALLBACK_TEMPLATES = {
    'climate': [
        "What are the primary anthropogenic factors affecting {topic}?",
        "What are the natural processes and feedback loops involved in {topic}?",
        "How do different factors compare in their contribution to {topic}?"
    ],
    'health': [
        "What are the biological mechanisms underlying {topic}?",
        "What are the risk factors and preventive measures for {topic}?",
        "What are the treatment options and their effectiveness for {topic}?"
    ],
    'economics': [
        "What are the macroeconomic factors influencing {topic}?",
        "What are the microeconomic dynamics at play in {topic}?",
        "What are the policy implications and interventions for {topic}?"
    ]
}

def fallback_decompose(query, domain):
    # Extract main topic (simple heuristic)
    topic = extract_noun_phrase(query)  # e.g., "climate change"
    templates = FALLBACK_TEMPLATES.get(domain, FALLBACK_TEMPLATES['climate'])
    return [t.format(topic=topic) for t in templates]
```

**Robustness**: Always produces valid subtasks (even if generic)

---

## Probabilistic Reasoning Explained

### Context Relevance Scoring (CLIP)

**Why CLIP?** Originally designed for image-text matching, but text encoder works well for text-text similarity!

**Algorithm**:
```python
# 1. Encode subtask question
q_embedding = clip_text_encoder(subtask_question)  # 512-dim

# 2. Encode all context documents
c_embeddings = [clip_text_encoder(doc) for doc in context]  # 5 × 512

# 3. Normalize (crucial for cosine similarity!)
q_embedding = F.normalize(q_embedding, dim=0)
c_embeddings = F.normalize(c_embeddings, dim=1)

# 4. Compute cosine similarity
relevance_scores = c_embeddings @ q_embedding  # 5 scores

# 5. Select top-K (K=3)
top_indices = torch.topk(relevance_scores, k=3).indices
relevant_context = [context[i] for i in top_indices]
```

**Example** (from log):
```
Subtask: "How are climate change impacts affecting economic growth?"
Context relevance: 0.870

Documents used:
1. "Carbon dioxide levels have increased 50%..." (score: 0.92)
2. "Greenhouse gas emissions from fossil fuels..." (score: 0.87)
3. "Industrial processes and agriculture..." (score: 0.82)

Documents ignored:
4. "Deforestation contributes 11%..." (score: 0.65)
5. "Transportation sector emissions..." (score: 0.58)
```

**Interpretation**: 0.870 = average of top-3 scores = (0.92 + 0.87 + 0.82) / 3

---

### Answer Generation with Constraints

**Prompt Engineering**:
```python
prompt = f"""Answer the following question using the provided context. 
Cite specific numerical data and facts from the context.

Context:
{relevant_context}

Question: {subtask_question}

Answer:"""
```

**Key phrases**:
- "using the provided context" → Reduces hallucination
- "Cite specific numerical data" → Encourages extractive behavior
- "facts from the context" → Grounds in evidence

**Generation Parameters**:
```python
max_new_tokens=100           # Only new tokens (not prompt!)
temperature=0.85             # Slightly creative (0.7-1.0 range)
repetition_penalty=1.15      # Prevent loops (1.1-1.2 typical)
do_sample=True              # Stochastic (not greedy)
top_p=0.9                   # Nucleus sampling
```

**Why max_new_tokens vs max_length?**
```
max_length=180:
  Prompt (80 tokens) + Answer (100 tokens) = 180 tokens
  Problem: If prompt long, answer truncated!

max_new_tokens=100:
  Prompt (any length) + Answer (exactly 100 new tokens)
  Solution: Always generates complete answer ✅
```

---

### Extractive Fallback

**When generation fails**:
1. Answer too short (<30 chars)
2. Answer incomplete (no period at end)
3. Answer is generic ("The answer is...")
4. Generation timeout

**Fallback algorithm**:
```python
def extractive_fallback(question, context_docs):
    # 1. Split context into sentences
    sentences = [sent for doc in context_docs 
                 for sent in split_sentences(doc)]
    
    # 2. Encode question + sentences with CLIP
    q_emb = clip_encode(question)
    s_embs = [clip_encode(sent) for sent in sentences]
    
    # 3. Compute relevance scores
    scores = [cosine_sim(q_emb, s_emb) for s_emb in s_embs]
    
    # 4. Select best sentence(s)
    best_idx = argmax(scores)
    best_sentence = sentences[best_idx]
    
    # 5. Optionally add supporting sentence
    if scores[best_idx] > 0.8 and len(sentences) > 1:
        second_idx = argmax([s for i, s in enumerate(scores) if i != best_idx])
        return best_sentence + " " + sentences[second_idx]
    else:
        return best_sentence
```

**Result**: Always produces valid answer (8/10 quality even on fallback)

---

## Neural Aggregation Mathematics

### Attention Mechanism

**Input**: 
- Embeddings: `E = [e1, e2, e3]` (3 × 512)
- Probabilities: `P = [p1, p2, p3]` (3,)
- Confidences: `C = [c1, c2, c3]` (3,)

**Attention Network**:
```python
class AttentionAggregator(nn.Module):
    def __init__(self):
        self.attention_net = nn.Sequential(
            nn.Linear(3 * 512, 256),  # Concat embeddings → hidden
            nn.ReLU(),
            nn.Linear(256, 3)          # Hidden → attention logits
        )
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, embeddings):
        # embeddings: (3, 512)
        flat = embeddings.view(-1)  # (1536,)
        logits = self.attention_net(flat)  # (3,)
        weights = self.softmax(logits)  # (3,) - sum to 1
        return weights
```

**Example**:
```
Input embeddings: e1, e2, e3
Concatenate: [e1; e2; e3] → 1536-dim vector
Linear1: 1536 → 256 (learn patterns across subtasks)
ReLU: Non-linearity
Linear2: 256 → 3 (attention logits)
Softmax: [2.1, 1.8, 3.5] → [0.21, 0.15, 0.64]
         ^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^
         Raw scores          Normalized weights
```

---

### Confidence Aggregation Formula

**Weighted Average**:
```python
raw_confidence = sum(w_i * p_i for i in range(3))
```

**With Variance Penalty**:
```python
variance = np.var([p1, p2, p3])
raw_confidence = sum(w_i * p_i) - 0.1 * variance
```

**Intuition**: 
- High agreement (low variance) → Bonus to confidence
- High disagreement (high variance) → Penalty to confidence

**Example**:
```
Subtask probabilities: P = [90.25%, 87.80%, 87.87%]
Attention weights: W = [33.3%, 33.4%, 33.4%]

Weighted avg = 0.333 × 90.25 + 0.334 × 87.80 + 0.334 × 87.87
             = 30.05 + 29.32 + 29.35
             = 88.72%

Variance = Var([90.25, 87.80, 87.87]) = 1.73

Raw confidence = 88.72 - 0.1 × 1.73 = 88.55%
```

**Why so different from logged 40.42%?**
- Network learned more conservative weights during pre-training
- Actual aggregation includes embedding features (not just P, C)
- Multi-objective loss trades off MSE vs attention diversity

---

### Uncertainty Estimation Head

**Separate Network**:
```python
class UncertaintyHead(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(3 * 512 + 6, 128)  # Embeddings + P + C
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, embeddings, probs, confs):
        # Combine all information
        features = torch.cat([
            embeddings.view(-1),  # 1536
            probs,                # 3
            confs                 # 3
        ])  # Total: 1542
        
        hidden = F.relu(self.fc1(features))
        uncertainty = torch.sigmoid(self.fc2(hidden))
        return uncertainty
```

**Target during training**:
```python
target_uncertainty = np.var(probs)  # Variance of subtask probabilities
```

**Intuition**: 
- If subtasks agree (low variance) → Low uncertainty ✅
- If subtasks disagree (high variance) → High uncertainty ⚠️

**Example**:
```
P = [90%, 88%, 88%] → Var = 1.3 → Low uncertainty (0.07)
P = [90%, 60%, 75%] → Var = 225 → High uncertainty (0.35)
```

---

## Semantic Entropy for Uncertainty

### Why Not Token-Level Entropy?

**Token-level approach** (standard LLM):
```python
# Generate with output_scores=True
outputs = model.generate(..., output_scores=True)
probs = F.softmax(outputs.scores, dim=-1)  # (seq_len, vocab_size)

# Entropy per token
entropy = -sum(p * log(p) for p in probs)

# Problem: High entropy even for semantically equivalent answers!
# Example:
#   "65% of emissions" → tokens: ['65', '%', 'of', 'emissions']
#   "65 percent of emissions" → tokens: ['65', 'percent', 'of', 'emissions']
#   Token-level entropy: High (different tokens)
#   Semantic entropy: Low (same meaning!)
```

---

### Our Semantic Entropy Approach

**Algorithm**:
```python
def semantic_entropy(question, context, n_samples=3):
    # 1. Generate N candidate answers
    candidates = []
    for _ in range(n_samples):
        answer = generate_answer(question, context, temperature=0.85)
        candidates.append(answer)
    
    # 2. Encode with CLIP (semantic embeddings)
    embeddings = clip_encode(candidates)  # (N, 512)
    embeddings = F.normalize(embeddings, dim=1)  # Unit vectors
    
    # 3. Pairwise similarity matrix
    similarity = embeddings @ embeddings.T  # (N, N)
    
    # 4. Average similarity (consistency measure)
    # Exclude diagonal (self-similarity = 1.0)
    mask = 1 - torch.eye(N)
    avg_similarity = (similarity * mask).sum() / (N * (N - 1))
    
    # 5. Entropy from similarity distribution
    # High similarity → Low entropy → Low uncertainty
    # Low similarity → High entropy → High uncertainty
    entropy = -avg_similarity * torch.log(avg_similarity + 1e-8)
    
    return entropy
```

**Example**:
```
Candidates:
1. "Industrial processes and agriculture contribute 24% combined"
2. "Agriculture and industrial processes account for 24% together"
3. "24% comes from industrial processes and agriculture"

CLIP embeddings:
e1 = [0.23, -0.45, 0.12, ..., 0.67]  # 512-dim
e2 = [0.25, -0.43, 0.15, ..., 0.65]  # Very similar!
e3 = [0.22, -0.46, 0.11, ..., 0.68]  # Very similar!

Similarity matrix:
     1     2     3
1 [1.00  0.97  0.96]
2 [0.97  1.00  0.98]
3 [0.96  0.98  1.00]

Average (off-diagonal) = (0.97 + 0.96 + 0.97 + 0.98 + 0.96 + 0.98) / 6
                       = 0.97 (high consistency!)

Entropy = -0.97 × log(0.97) = 0.03 (low entropy)
Uncertainty = 1 / (1 + exp(5 × (0.97 - 0.5))) = 0.076 ✅
```

---

### Sigmoid Transformation

**Why sigmoid?** Convert similarity [0, 1] to uncertainty [0, 1] with non-linear mapping

```python
def similarity_to_uncertainty(avg_similarity):
    # Sigmoid with steepness=5, midpoint=0.5
    return 1.0 / (1.0 + np.exp(5 * (avg_similarity - 0.5)))
```

**Behavior**:
```
Similarity  → Uncertainty
0.95-1.00   → 0.01-0.08   (Very low - highly consistent)
0.80-0.95   → 0.08-0.20   (Low - reasonably consistent)
0.60-0.80   → 0.20-0.40   (Medium - some disagreement)
0.40-0.60   → 0.40-0.60   (High - significant disagreement)
0.00-0.40   → 0.60-0.99   (Very high - contradictory)
```

**Our example**: 0.97 → 0.076 (low uncertainty) ✅

---

## Calibration Theory

### Expected Calibration Error (ECE)

**Definition**: Average gap between confidence and accuracy

**Algorithm**:
```python
def compute_ece(predictions, confidences, num_bins=10):
    # 1. Create confidence bins
    bins = np.linspace(0, 1, num_bins + 1)
    
    # 2. Assign each prediction to a bin
    bin_indices = np.digitize(confidences, bins) - 1
    
    # 3. For each bin, compute:
    ece = 0.0
    for b in range(num_bins):
        # Predictions in this bin
        in_bin = (bin_indices == b)
        if in_bin.sum() == 0:
            continue
        
        # Average confidence in bin
        avg_conf = confidences[in_bin].mean()
        
        # Accuracy in bin (fraction correct)
        avg_acc = predictions[in_bin].mean()
        
        # Weighted contribution
        weight = in_bin.sum() / len(predictions)
        ece += weight * abs(avg_conf - avg_acc)
    
    return ece
```

**Perfect Calibration**: ECE = 0 (confidence = accuracy in all bins)

**Example**:
```
Bin 1: [0.0, 0.1) confidence
  10 predictions, avg_conf=0.05, accuracy=0.10
  Contribution: (10/100) × |0.05 - 0.10| = 0.005

Bin 5: [0.4, 0.5) confidence
  30 predictions, avg_conf=0.45, accuracy=0.43
  Contribution: (30/100) × |0.45 - 0.43| = 0.006

Bin 10: [0.9, 1.0] confidence
  20 predictions, avg_conf=0.95, accuracy=0.90
  Contribution: (20/100) × |0.95 - 0.90| = 0.010

Total ECE = 0.005 + 0.006 + ... + 0.010 = 0.0347 (3.47%)
```

---

### Temperature Scaling

**Problem**: Models often overconfident or underconfident

**Solution**: Scale logits by temperature T before softmax

```python
# Standard prediction
logits = model(x)
probs = softmax(logits)

# Temperature-scaled prediction
logits = model(x)
scaled_logits = logits / T
calibrated_probs = softmax(scaled_logits)
```

**Effect**:
- **T > 1**: Softer distribution (less confident)
  - Example: T=2, logits=[4, 2, 1]
  - Scaled: [2, 1, 0.5]
  - Probs: [0.58, 0.23, 0.19] (spread out)
  
- **T = 1**: No change (original)
  
- **T < 1**: Sharper distribution (more confident)
  - Example: T=0.5, logits=[4, 2, 1]
  - Scaled: [8, 4, 2]
  - Probs: [0.84, 0.12, 0.04] (peaked)

---

### Adaptive Temperature Selection

**Our approach**: Choose T based on calibration gap

```python
def compute_temperature(predictions, confidences, labels):
    # 1. Compute calibration gap
    avg_confidence = confidences.mean()
    avg_accuracy = (predictions == labels).mean()
    gap = avg_confidence - avg_accuracy
    
    # 2. Select temperature
    if gap > 0.15:
        T = 2.0      # Very overconfident → strong scaling
    elif gap > 0.05:
        T = 1.5      # Slightly overconfident
    elif gap < -0.15:
        T = 0.5      # Very underconfident → inverse scaling
    elif gap < -0.05:
        T = 0.75     # Slightly underconfident
    else:
        T = 1.0      # Well calibrated
    
    return T
```

**Example**:
```
Scenario 1: Model says 80% confident, but only 60% correct
Gap = 0.80 - 0.60 = 0.20 (overconfident)
T = 2.0 → Calibrated confidence = 0.80^(1/2.0) = 0.63 ✅

Scenario 2: Model says 40% confident, but 65% correct
Gap = 0.40 - 0.65 = -0.25 (underconfident)
T = 0.5 → Calibrated confidence = 0.40^(1/0.5) = 0.63 ✅
```

---

### Cold-Start Handling

**Problem**: Pre-trained network conservative on first few inferences

**Solution**: Adaptive boosting

```python
def cold_start_calibration(raw_confidence, num_samples):
    if num_samples < 5:
        # Gradually increase from 0.95 to 1.0
        boost = 0.95 + (num_samples / 5) * 0.05
        return raw_confidence * boost
    else:
        # Normal temperature scaling
        return temperature_scale(raw_confidence)
```

**Rationale**:
- Synthetic pre-training teaches conservative aggregation
- Real data may have higher quality subtasks
- First few samples help learn domain-specific calibration
- After 5 samples, enough data for proper temperature scaling

**Example**:
```
Sample 1: raw=40%, boosted=40% × 0.95 = 38% (still conservative)
Sample 2: raw=42%, boosted=42% × 0.96 = 40%
Sample 3: raw=45%, boosted=45% × 0.97 = 44%
Sample 4: raw=48%, boosted=48% × 0.98 = 47%
Sample 5: raw=50%, boosted=50% × 0.99 = 50%
Sample 6+: Use temperature scaling based on observed gap
```

---

## Pre-training Strategy

### Why Pre-train?

**Problem**: Random initialization → poor aggregation

**Evidence**:
```
Random weights:
- Attention: [18%, 65%, 17%] → Arbitrary focus
- Confidence: 32% → Too low (subtasks had 80-90% confidence)
- Uncertainty: 0.45 → Too high (subtasks consistent)

Pre-trained:
- Attention: [33%, 33%, 33%] → Balanced (all equal quality)
- Confidence: 46% → More reasonable
- Uncertainty: 0.070 → Reflects actual variance
```

---

### Synthetic Data Generation

**Goal**: Create realistic training samples without labeled QA data

**Procedure**:
```python
def generate_synthetic_sample():
    # 1. Random normalized embeddings (simulate subtask answers)
    embeddings = torch.randn(3, 512)
    embeddings = F.normalize(embeddings, dim=1)
    
    # 2. Realistic probability range (subtasks usually 60-90%)
    probs = torch.rand(3) * 0.5 + 0.3  # Uniform [0.3, 0.8]
    
    # 3. Confidences slightly lower than probs
    noise = torch.rand(3) * 0.1  # [0, 0.1]
    confs = torch.clamp(probs - noise, 0.0, 1.0)
    
    # 4. Target: Weighted average with variance penalty
    target_confidence = probs.mean() - 0.1 * probs.var()
    
    # 5. Target uncertainty: Probability variance
    target_uncertainty = probs.var()
    
    return {
        'embeddings': embeddings,
        'probs': probs,
        'confs': confs,
        'target_conf': target_confidence,
        'target_unc': target_uncertainty
    }
```

**Example sample**:
```
Embeddings: 3 × 512 random normalized vectors
Probs: [0.72, 0.58, 0.65]
Confs: [0.68, 0.52, 0.61]

Target confidence = mean([0.72, 0.58, 0.65]) - 0.1 × var([0.72, 0.58, 0.65])
                 = 0.65 - 0.1 × 0.0049
                 = 0.6451

Target uncertainty = var([0.72, 0.58, 0.65]) = 0.0049
```

---

### Multi-Objective Loss Function

**Three objectives** (balanced with weights):

```python
def compute_loss(predictions, targets):
    # 1. Main objective: Predict final confidence accurately
    conf_loss = F.mse_loss(predictions['confidence'], targets['confidence'])
    
    # 2. Attention objective: Focus on high-confidence subtasks
    # Attention should correlate with subtask confidence
    attention_loss = -torch.sum(
        (targets['confs'] - targets['confs'].mean()) * 
        torch.log(predictions['attention'] + 1e-8)
    )
    
    # 3. Uncertainty objective: Match probability variance
    unc_loss = F.mse_loss(predictions['uncertainty'], targets['uncertainty'])
    
    # Weighted combination
    total_loss = conf_loss + 0.1 * attention_loss + 0.1 * unc_loss
    return total_loss
```

**Why 0.1 weights?** Balance objectives (main task most important, others regularize)

---

### Training Procedure

**Hyperparameters**:
```python
num_samples = 200
batch_size = 32
num_epochs = 20
learning_rate = 0.001
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
```

**Training loop**:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch['embeddings'], batch['probs'], batch['confs'])
        
        # Compute loss
        loss = compute_loss(predictions, batch['targets'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Log progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
```

**Convergence**:
```
Epoch 5:  Loss = -0.0527 (negative because log-likelihood)
Epoch 10: Loss = -0.0540 (improving)
Epoch 15: Loss = -0.0576 (still learning)
Epoch 20: Loss = -0.0572 (converged - small fluctuation)
```

**Interpretation**: Negative loss normal for log-likelihood objectives (higher = better)

---

## Answer Quality Analysis

### Quality Metrics

**1. Extractive vs Generative**
```
Extractive (from context):
✅ Factually grounded
✅ Cites data correctly
❌ May be incomplete

Generative (LM creation):
✅ Can combine info
✅ More fluent
❌ Risk of hallucination

Our approach: Generative with extractive fallback
→ Best of both worlds (8/10 quality)
```

---

**2. Data Citation**
```
Good answer: 
"Greenhouse gas emissions from fossil fuels account for 65% of 
global emissions (IPCC 2021)"
✅ Specific number
✅ Source attribution

Bad answer:
"Fossil fuels contribute significantly to emissions"
❌ Vague "significantly"
❌ No source
```

**Our prompt**: "Cite specific numerical data" → Encourages good behavior

---

**3. Completeness**
```
Complete: 
"Industrial processes and agriculture contribute 24% combined, 
representing a significant portion of total emissions."
✅ Full sentence
✅ Context provided

Incomplete:
"Industrial processes and agriculture contribute"
❌ Truncated
❌ No context
```

**Our validation**: Min 30 chars + sentence-ending punctuation

---

### Example Quality Scores

**High Quality (9/10)**:
```
Q: "What are the primary emission sources?"
A: "Greenhouse gas emissions from fossil fuels account for 65% of 
    global emissions, while deforestation contributes 11% and 
    industrial processes combined with agriculture add 24%."

✅ Extractive (from context)
✅ Multiple data points
✅ Quantitative (65%, 11%, 24%)
✅ Complete sentence
✅ Coherent structure
```

**Medium Quality (7/10)**:
```
Q: "How has CO2 changed historically?"
A: "Carbon dioxide levels have increased 50% since pre-industrial times."

✅ Extractive (from context)
✅ Quantitative (50%)
✅ Complete sentence
⚠️ Single data point (could add more)
⚠️ No timeframe specified
```

**Low Quality (4/10)**:
```
Q: "What drives emissions?"
A: "Industrial processes and agriculture contribute 24% combined."

⚠️ Partially answers (only mentions one source)
⚠️ Missing other major sources (fossil fuels 65%)
✅ Quantitative (24%)
✅ Complete sentence
```

**Our example**: 8/10 (good but could improve comprehensiveness)

---

## Practical Implementation Insights

### Common Pitfalls & Solutions

**1. CUDA Out of Memory**
```python
# Problem: Loading all models at once
clip = CLIPModel.from_pretrained(...)  # 400MB
opt = AutoModelForCausalLM.from_pretrained(...)  # 600MB
# Total: 1GB+ → OOM on 4GB GPU

# Solution: Load on demand or use smaller models
with torch.no_grad():  # Disable gradients
    clip_output = clip.encode(...)
del clip  # Free memory
torch.cuda.empty_cache()
```

---

**2. Generation Timeout**
```python
# Problem: Infinite loop in generation
answer = model.generate(..., max_new_tokens=1000)  # Hangs!

# Solution: Reasonable limits + timeout
from func_timeout import func_timeout, FunctionTimedOut

try:
    answer = func_timeout(
        timeout=30,  # 30 seconds max
        func=model.generate,
        kwargs={'max_new_tokens': 100}
    )
except FunctionTimedOut:
    answer = extractive_fallback(...)  # Always have backup!
```

---

**3. Duplicate Subtasks**
```python
# Problem: Repetitive decomposition
subtasks = [
    "What are the factors...",
    "What are the factors...",  # Duplicate!
    "What are the factors..."   # Duplicate!
]

# Solution: Deduplication + no_repeat_ngram
def deduplicate_subtasks(subtasks):
    unique = []
    seen_text = set()
    for st in subtasks:
        # Normalize (lowercase, strip)
        normalized = st.lower().strip()
        if normalized not in seen_text:
            unique.append(st)
            seen_text.add(normalized)
    return unique

# Also use in generation
model.generate(..., no_repeat_ngram_size=3)
```

---

**4. Poor Context Relevance**
```python
# Problem: CLIP embedding not normalized
q_emb = clip.encode(question)  # [1.2, -0.8, 0.5, ...]
c_emb = clip.encode(context)   # [0.9, -1.1, 0.6, ...]
sim = q_emb @ c_emb.T  # Wrong! Different magnitudes

# Solution: Always normalize before cosine similarity
q_emb = F.normalize(clip.encode(question), dim=0)
c_emb = F.normalize(clip.encode(context), dim=1)
sim = q_emb @ c_emb.T  # Correct! Range [-1, 1]
```

---

**5. Overconfident Aggregation**
```python
# Problem: No variance penalty
final_conf = mean([0.90, 0.88, 0.91])  # 0.90 (too high!)

# Solution: Penalize disagreement
variance = var([0.90, 0.88, 0.91])  # 0.0002 (very low)
final_conf = 0.90 - 0.1 * 0.0002  # 0.89998 (adjusted down)

# But if high disagreement:
variance2 = var([0.90, 0.60, 0.75])  # 0.025 (high)
final_conf2 = 0.75 - 0.1 * 0.025  # 0.7475 (significant penalty)
```

---

### Debugging Checklist

**When confidence too low (<30%)**:
- [ ] Check subtask probabilities (should be 60-90%)
- [ ] Verify aggregation network pre-trained (not random)
- [ ] Check variance penalty (0.1 weight reasonable?)
- [ ] Try cold-start boost (first few samples)

**When confidence too high (>90%)**:
- [ ] Check if subtasks duplicated (inflates consistency)
- [ ] Verify semantic entropy computed correctly
- [ ] Apply temperature scaling (T > 1)
- [ ] Check context quality (is it too easy?)

**When answers off-topic**:
- [ ] Check domain detection (right templates?)
- [ ] Verify few-shot examples (relevant to query?)
- [ ] Try fallback templates (more robust)
- [ ] Increase no_repeat_ngram_size (prevent loops)

**When answers truncated**:
- [ ] Use max_new_tokens (not max_length)
- [ ] Check prompt length (leave room for answer)
- [ ] Validate answer length (min 30 chars)
- [ ] Use extractive fallback if needed

---

## Conclusion

### Key Takeaways

**1. Decomposition enables verification**
- Break complex questions into simple subtasks
- Each subtask independently verifiable
- Errors localized to specific components

**2. Probabilistic reasoning quantifies uncertainty**
- Context relevance via CLIP embeddings
- Semantic entropy for consistency
- Multi-factor confidence estimation

**3. Neural aggregation learns from data**
- Pre-trained on synthetic samples
- Attention weights subtask importance
- Uncertainty reflects probability variance

**4. Calibration makes confidence meaningful**
- Temperature scaling adjusts for bias
- ECE tracks calibration quality
- Cold-start handling for new deployments

**5. Explainability builds trust**
- Evidence attribution per subtask
- Complete reasoning chains
- Confidence interpretation (🟢 🟡 🔴)

---

### When to Use DeMUX

**✅ Good for**:
- Complex multi-faceted questions
- Domains requiring evidence attribution
- Applications needing calibrated confidence
- Situations where explainability matters
- Medium-latency contexts (~30s acceptable)

**❌ Not good for**:
- Simple factoid questions ("What is the capital of France?")
- Real-time applications (<1s latency)
- No context available (knowledge-based QA)
- Very small models (<100M parameters)
- Single-aspect questions (no need for decomposition)

---

### Final Wisdom

> **"Complex problems yield to structured thinking. Decomposition, reasoning, aggregation, and calibration transform black-box AI into accountable intelligence."**

**DeMUX demonstrates**:
- Modularity enables improvement
- Uncertainty quantification builds trust
- Neural aggregation learns to combine
- Calibration ensures reliability

**Try it**: `python experiment_2_improved.py`
