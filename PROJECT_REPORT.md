# Recursive vs. Iterative Minimax Ghost – Project Report

## 1. Project Overview

This project extends the UC Berkeley CS188 Pacman framework by implementing **two minimax ghost agents** that use adversarial search to pursue Pacman intelligently. The key modification compares **recursive** vs. **iterative** implementations of the same algorithm to analyze correctness equivalence and performance characteristics.

| Aspect | Detail |
|---|---|
| **Base Framework** | CS188 Multiagent Pacman |
| **Core Task** | Ghosts use Minimax (depth 2) against Reflex Pacman |
| **Version A** | Recursive minimax with alpha-beta pruning |
| **Version B** | Iterative minimax using explicit stack frames |
| **Evaluation** | 50 games per version, recording win rate, decision time, and depth/stack metrics |

---

## 2. System Architecture

```mermaid
graph TB
    subgraph "Pacman Game Engine"
        P["pacman.py<br/>Game Loop & CLI"]
        G["game.py<br/>Game State & Rules"]
        L["layouts/<br/>Map Definitions"]
    end

    subgraph "Agent Layer"
        PA["pacmanAgents.py<br/>ReflexAgent / GreedyAgent"]
        GA["ghostAgents.py<br/>Ghost Agent Classes"]
    end

    subgraph "Minimax Ghost Agents"
        EVAL["_ghostEvaluation()<br/>Score + Distance Heuristic"]
        REC["MinimaxGhostRecursive<br/>Version A"]
        ITER["MinimaxGhostIterative<br/>Version B"]
    end

    subgraph "Benchmarking"
        BM["benchmark.py<br/>50-Game Evaluation"]
        OUT["results.json<br/>Output Report"]
    end

    P --> G
    P --> L
    P --> PA
    P --> GA
    GA --> REC
    GA --> ITER
    REC --> EVAL
    ITER --> EVAL
    BM --> REC
    BM --> ITER
    BM --> OUT

    style REC fill:#2d6a4f,stroke:#1b4332,color:#fff
    style ITER fill:#1d3557,stroke:#0d1b2a,color:#fff
    style EVAL fill:#e76f51,stroke:#d62828,color:#fff
    style BM fill:#7209b7,stroke:#560bad,color:#fff
```

---

## 3. How the Algorithms Work

### 3.1 The Minimax Idea

Minimax is a decision strategy for **two-player zero-sum games**. One player (MAX) tries to **maximize** the outcome, the other (MIN) tries to **minimize** it. Each player assumes the opponent plays **optimally**.

In our Pacman setup:
- **Ghost (self)** = **MAX** — wants the highest evaluation (Pacman dying)
- **Pacman + other ghosts** = **MIN** — assumed to play the worst case for our ghost

The algorithm builds a **game tree** by simulating future moves, then picks the action that leads to the best guaranteed outcome.

### 3.2 Concrete Game Tree Example

Imagine a simplified scenario: Ghost has 2 moves (Left, Right), then Pacman responds with 2 moves each. The leaf nodes are evaluation scores from the ghost's perspective:

```mermaid
graph TD
    ROOT["🔴 Ghost - MAX<br/>Pick highest"]

    ROOT -->|"Left"| P1["🔵 Pacman - MIN<br/>Pick lowest"]
    ROOT -->|"Right"| P2["🔵 Pacman - MIN<br/>Pick lowest"]

    P1 -->|"Up"| L1["Eval: 3 ✓"]
    P1 -->|"Down"| L2["Eval: 5"]

    P2 -->|"Up"| L3["Eval: 2 ✓"]
    P2 -->|"Down"| L4["Eval: 8"]

    style ROOT fill:#2d6a4f,stroke:#1b4332,color:#fff
    style P1 fill:#1d3557,stroke:#0d1b2a,color:#fff
    style P2 fill:#1d3557,stroke:#0d1b2a,color:#fff
    style L1 fill:#e76f51,stroke:#d62828,color:#fff
    style L3 fill:#e76f51,stroke:#d62828,color:#fff
```

**Step by step:**

| Step | Node | Logic | Result |
|---|---|---|---|
| 1 | Pacman after Ghost-Left | MIN picks min(3, 5) | **3** |
| 2 | Pacman after Ghost-Right | MIN picks min(2, 8) | **2** |
| 3 | Ghost (root) | MAX picks max(3, 2) | **3 → choose Left** |

> The ghost picks **Left** (guaranteed score 3) because even though Right *could* lead to 8, a smart Pacman (MIN) would choose the move giving 2 instead.

### 3.3 Alpha-Beta Pruning — Skipping Unnecessary Work

Without pruning, minimax explores **every leaf**. Alpha-beta introduces two bounds:
- **α (alpha)** — best score MAX can guarantee so far (starts at −∞)
- **β (beta)** — best score MIN can guarantee so far (starts at +∞)

> **Key rule: when α ≥ β, PRUNE** — stop exploring that branch because the result won't affect the final decision.

#### Step-by-step walkthrough with pruning:

```mermaid
graph TD
    ROOT["🔴 Ghost - MAX<br/>α=−∞, β=+∞"]

    ROOT -->|"1️⃣ Left"| P1["🔵 Pacman - MIN"]
    ROOT -->|"2️⃣ Right"| P2["🔵 Pacman - MIN"]

    P1 -->|"a"| L1["Eval: 3"]
    P1 -->|"b"| L2["Eval: 5"]

    P2 -->|"c"| L3["Eval: 2"]
    P2 -->|"d ✂️"| L4["PRUNED!"]

    style ROOT fill:#2d6a4f,stroke:#1b4332,color:#fff
    style P1 fill:#1d3557,stroke:#0d1b2a,color:#fff
    style P2 fill:#1d3557,stroke:#0d1b2a,color:#fff
    style L4 fill:#d62828,stroke:#9d0208,color:#fff
```

| Step | What happens | α | β | Note |
|---|---|---|---|---|
| 1 | Explore leaf **a** → val=3 | −∞ | 3 | P1 sets β=min(∞,3)=3 |
| 2 | Explore leaf **b** → val=5 | −∞ | 3 | P1: min(3,5)=3, β stays 3 |
| 3 | P1 returns **3** to ROOT | **3** | +∞ | ROOT sets α=max(−∞,3)=**3** |
| 4 | Explore leaf **c** → val=2 | 3 | **2** | P2 sets β=min(∞,2)=2 |
| 5 | **α(3) ≥ β(2) → PRUNE** ✂️ | 3 | 2 | Skip leaf **d** entirely! |
| 6 | P2 returns 2. ROOT: max(3,2)=3 | 3 | +∞ | Best action = **Left** |

> We **never evaluated leaf d** (value 8) because we already knew the Right subtree couldn't beat score 3. This saves computation without changing the result!

### 3.4 Multi-Agent Depth Counting

In Pacman, there are **multiple agents** (Pacman + N ghosts). A "depth" of 2 means the controlling ghost looks **2 full rounds ahead**, where each round has every agent move once:

```mermaid
graph TD
    subgraph "Depth 2 — full round"
        G1_2["Ghost 1 MAX"]
        G1_2 --> P_2["Pacman MIN"]
        P_2 --> G2_2["Ghost 2 MIN"]
    end

    subgraph "Depth 1 — full round"
        G2_2 --> G1_1["Ghost 1 MAX"]
        G1_1 --> P_1["Pacman MIN"]
        P_1 --> G2_1["Ghost 2 MIN"]
    end

    G2_1 --> LEAF["evalFn — depth 0"]

    style G1_2 fill:#2d6a4f,stroke:#1b4332,color:#fff
    style G1_1 fill:#2d6a4f,stroke:#1b4332,color:#fff
    style P_2 fill:#d62828,stroke:#9d0208,color:#fff
    style P_1 fill:#d62828,stroke:#9d0208,color:#fff
    style G2_2 fill:#e76f51,stroke:#d62828,color:#fff
    style G2_1 fill:#e76f51,stroke:#d62828,color:#fff
    style LEAF fill:#7209b7,stroke:#560bad,color:#fff
```

In code, depth only decrements when control **wraps back** to the MAX ghost:
```python
nextAgent = (agentIndex + 1) % numAgents
nextDepth = depth - 1 if nextAgent == self.index else depth
```

### 3.5 Version A — Recursive Implementation

The recursive version is the natural, textbook way to write minimax. Python's **call stack** handles the tree navigation automatically:

```mermaid
graph TD
    START["_minimax(state, depth, agent, α, β)"] --> TERM{"Terminal?<br/>win/lose/depth=0"}
    TERM -->|Yes| EVAL["Return evalFn(state)"]
    TERM -->|No| CHECK{"agent == self?"}
    CHECK -->|"Yes = MAX"| MAX_LOOP["For each action:<br/>val = minimax(successor, ...)"]
    CHECK -->|"No = MIN"| MIN_LOOP["For each action:<br/>val = minimax(successor, ...)"]
    MAX_LOOP --> ALPHA{"val > β?"}
    ALPHA -->|"Yes"| PRUNE_B["β cut-off ✂️<br/>BREAK"]
    ALPHA -->|"No"| UPD_A["α = max(α, val)<br/>next action"]
    MIN_LOOP --> BETA{"val < α?"}
    BETA -->|"Yes"| PRUNE_A["α cut-off ✂️<br/>BREAK"]
    BETA -->|"No"| UPD_B["β = min(β, val)<br/>next action"]

    style START fill:#2d6a4f,stroke:#1b4332,color:#fff
    style PRUNE_A fill:#d62828,stroke:#9d0208,color:#fff
    style PRUNE_B fill:#d62828,stroke:#9d0208,color:#fff
    style EVAL fill:#e76f51,stroke:#d62828,color:#fff
```

**Simplified code:**
```python
def _minimax(self, state, depth, agentIndex, alpha, beta):
    # Base case
    if state.isWin() or state.isLose() or depth == 0:
        return self.evalFn(state, self.index), None

    nextAgent = (agentIndex + 1) % numAgents
    nextDepth = depth - 1 if nextAgent == self.index else depth

    if agentIndex == self.index:       # MAX node (this ghost)
        bestValue = float('-inf')
        for action in legalActions:
            val, _ = self._minimax(successor, nextDepth, nextAgent, alpha, beta)
            if val > bestValue:
                bestValue, bestAction = val, action
            alpha = max(alpha, bestValue)
            if bestValue > beta: break             # β cut-off ✂️
    else:                              # MIN node (pacman / other ghost)
        bestValue = float('inf')
        for action in legalActions:
            val, _ = self._minimax(successor, nextDepth, nextAgent, alpha, beta)
            if val < bestValue:
                bestValue, bestAction = val, action
            beta = min(beta, bestValue)
            if bestValue < alpha: break            # α cut-off ✂️

    return bestValue, bestAction
```

**Pros:** Simple, readable, direct translation of the algorithm.
**Cons:** Limited by Python's recursion depth (~1000 calls).

### 3.6 Version B — Iterative Implementation with Explicit Stack

The iterative version replaces Python's call stack with a **list of dictionaries**, where each dict is a "stack frame". This requires a two-phase state machine:

```mermaid
stateDiagram-v2
    [*] --> EXPAND: Push root frame
    EXPAND --> POP_TERMINAL: Terminal state
    POP_TERMINAL --> [*]: Stack empty
    POP_TERMINAL --> COLLECT: Parent exists
    EXPAND --> COLLECT: Setup actions + push first child
    COLLECT --> EXPAND: Push next child
    COLLECT --> POP_DONE: All actions done or pruned
    POP_DONE --> [*]: Stack empty
    POP_DONE --> COLLECT: Parent exists
```

**Each stack frame stores the equivalent of local variables:**
```python
frame = {
    'state':      GameState,   # what recursive call receives as arg
    'depth':      int,         # remaining search depth
    'agentIndex': int,         # whose turn it is
    'alpha':      float,       # α bound (copied from parent)
    'beta':       float,       # β bound (copied from parent)
    'isMax':      bool,        # MAX or MIN node
    'actions':    list,        # legal actions to try
    'actionIdx':  int,         # which action we're currently on
    'bestValue':  float,       # best value seen so far
    'bestAction': str,         # action that gave best value
    'phase':      EXPAND|COLLECT  # which phase we're in
}
```

**How the phases work:**

```mermaid
graph TD
    subgraph "EXPAND Phase"
        E1["Check if terminal → if yes, pop and set returnValue"]
        E2["Get legal actions"]
        E3["Initialize bestValue, bestAction"]
        E4["Switch phase to COLLECT"]
        E5["Push first child frame"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph "COLLECT Phase"
        C1["Read returnValue from child"]
        C2["Update bestValue if child is better"]
        C3["Update α or β"]
        C4{"Pruned or no more actions?"}
        C4 -->|"No"| C5["Push next child frame"]
        C4 -->|"Yes"| C6["Pop frame, set returnValue for parent"]
        C1 --> C2 --> C3 --> C4
    end

    E5 -.->|"child runs"| C1

    style E1 fill:#1d3557,stroke:#0d1b2a,color:#fff
    style C6 fill:#e76f51,stroke:#d62828,color:#fff
```

**The key translation rules:**

| Recursive concept | Iterative equivalent |
|---|---|
| Making a recursive call | Push a new frame dict onto the stack |
| Returning a value | Pop the frame, store result in `returnValue` |
| Local variables (`bestValue`, etc.) | Fields in the frame dictionary |
| Passing `α, β` as arguments | Copy from parent frame when pushing child |
| For-loop over actions | `actionIdx` counter incremented in COLLECT |
| Base case check | Done at EXPAND phase → pop immediately |

**Pros:** No recursion limit — can search arbitrarily deep.
**Cons:** ~7% slower due to dict creation overhead; more complex code.

### 3.7 Why Both Produce Identical Moves

Since the iterative version is a **mechanical translation** of the recursive one, they traverse the game tree in **exactly the same order** and apply **the same pruning rules** at every node. This guarantees identical action selection, which we verified empirically:

```
$ python benchmark.py --test-equivalence --layout testClassic
  ✓ PASS – 72 rounds, all ghost moves identical.
```

---

## 4. Evaluation Function

```mermaid
graph LR
    STATE["Game State"] --> NEG["-getScore()"]
    STATE --> LOSE{"isLose?"}
    STATE --> WIN{"isWin?"}
    NEG --> SUM["score"]
    LOSE -->|"+500"| SUM
    WIN -->|"-500"| SUM
    SUM --> FINAL["Return score"]

    style FINAL fill:#e76f51,stroke:#d62828,color:#fff
```

The ghost evaluation **negates** Pacman's score (lower Pacman score = better for ghost), with terminal bonuses:

```python
def _ghostEvaluation(state, ghostIndex):
    score = -state.getScore()
    if state.isLose(): score += 500   # Ghost wins → big bonus
    if state.isWin():  score -= 500   # Ghost loses → big penalty
    return score
```

---

## 5. Benchmark Results

### Configuration
- **Layout**: `originalClassic` | **Ghosts**: 2 | **Depth**: 2 | **Pacman**: ReflexAgent

### Results Summary

| Metric | Version A (Recursive) | Version B (Iterative) |
|---|---|---|
| Ghost Win Rate | 100% | 100% |
| Pacman Win Rate | 0% | 0% |
| Avg Pacman Score | 249.0 | 249.0 |
| Avg Decision Time | 8.518 ms | 9.188 ms |
| Max Recursion Depth / Stack Size | 6 | 7 |
| Decisions per Game | 502 | 502 |

### Performance Comparison

```mermaid
xychart-beta
    title "Decision Time Comparison (ms)"
    x-axis ["Recursive (A)", "Iterative (B)"]
    y-axis "Avg Decision Time (ms)" 0 --> 12
    bar [8.518, 9.188]
```

```mermaid
xychart-beta
    title "Max Tree Depth / Stack Size"
    x-axis ["Recursive (A)", "Iterative (B)"]
    y-axis "Max Depth / Stack Size" 0 --> 10
    bar [6, 7]
```

### Key Observations

1. **Behavioral Equivalence** ✅ — Both versions produce identical moves, verified on multiple layouts
2. **Recursive is ~7% faster** — Lower per-call overhead (no dict creation for stack frames)
3. **Iterative uses +1 stack level** — The explicit stack counts the root frame, while recursion depth starts at 0
4. **Trade-off**: Recursive is simpler and faster; Iterative avoids Python's recursion limit for deep searches

---

## 6. File Structure

```mermaid
graph TD
    subgraph "Modified Files"
        GA["ghostAgents.py<br/>+ MinimaxGhostRecursive<br/>+ MinimaxGhostIterative<br/>+ _ghostEvaluation()"]
    end

    subgraph "New Files"
        BM["benchmark.py<br/>50-game evaluation<br/>--test-equivalence<br/>--display / --output"]
    end

    subgraph "Unmodified CS188 Framework"
        PM["pacman.py"]
        GM["game.py"]
        MA["multiAgents.py"]
        UT["util.py"]
        LAY["layouts/"]
    end

    PM -->|"-g flag"| GA
    BM --> GA
    BM --> PM

    style GA fill:#2d6a4f,stroke:#1b4332,color:#fff
    style BM fill:#7209b7,stroke:#560bad,color:#fff
```

---

## 7. Usage Guide

```bash
# Run with recursive ghost (Version A)
python pacman.py -p GreedyAgent -g MinimaxGhostRecursive -l smallClassic

# Run with iterative ghost (Version B)
python pacman.py -p GreedyAgent -g MinimaxGhostIterative -l smallClassic

# Full 50-game benchmark with JSON output
python benchmark.py --num-games 50 --layout smallClassic -o results.json

# Verify both versions produce identical moves
python benchmark.py --test-equivalence

# Watch games visually
python benchmark.py --display --frame-time 0.05 --num-games 5

# Customize ghosts and depth
python benchmark.py --num-ghosts 2 --depth 3 --layout mediumClassic
```
