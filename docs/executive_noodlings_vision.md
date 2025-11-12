# Executive Noodlings: Stateful AI for Programming Tasks

**Date**: November 6, 2025
**Status**: Vision Document
**Authors**: Caity & Claude

## Executive Summary

Current AI programming assistants operate statelessly, processing each interaction as an isolated event with expanding context windows. This document proposes adapting the Noodlings architecture—originally designed for affective social agents—to programming and executive tasks. The core insight: **temporal hierarchies, surprise-driven learning, and selective memory aren't just for emotions; they're general principles of situated cognition.**

## The Problem with Current Approaches

### Stateless AI Programming Tools

Most AI coding assistants follow this pattern:
```
User Query → Load Full Context → LLM Processing → Response → Discard State
```

**Limitations:**
- No persistent understanding between sessions
- Context window grows until it hits limits
- No learning from project-specific patterns
- "Forgets" previous architectural decisions
- Expensive token costs for repeated context loading
- No intuition about what "feels wrong" in a codebase

### "Plan Mode" Isn't Enough

Plan/architect modes in current tools are essentially prompt engineering:
- Still stateless underneath
- No persistent memory of past plans
- Can't learn project-specific patterns
- No temporal reasoning about long-term vs. immediate concerns

## The Noodlings Architecture: A Different Approach

### Current Design (Affective/Social)

```
Fast Layer (LSTM, 16-D):    Immediate reactions, conversation flow
Medium Layer (LSTM, 16-D):  Recent interaction patterns
Slow Layer (GRU, 8-D):      Personality, long-term relationships
                            ↓
                     Predictor Network
                            ↓
                   Surprise Metric (L2 distance)
                            ↓
                  Episodic Memory Retrieval
                            ↓
                    LLM Context Building
```

**Key Properties:**
1. **Hierarchical temporal processing**: Different layers for different timescales
2. **Surprise-driven attention**: Notice when things deviate from expectations
3. **Selective retrieval**: Fetch relevant memories, not everything
4. **Continuous state**: Maintains understanding across interactions
5. **Learning**: Updates internal models based on experience

### Adapted for Programming (Executive Noodlings)

```
Fast Layer:     Current file/function context, immediate syntax
Medium Layer:   Active task flow, related code sections
Slow Layer:     Project architecture, coding patterns, tech stack
                            ↓
                     Predictor Network
                            ↓
               Surprise: Code/behavior mismatches
                            ↓
            Retrieval: Relevant code/docs/patterns
                            ↓
                    LLM Context Building
```

## Core Architectural Innovations

### 1. Persistent Project Understanding

**Current Tools:**
```python
# Session 1
"Create a React component with Redux"
[Generates component]

# Session 2 (next day)
"Add another component"
[Doesn't remember we use Redux, asks for clarification]
```

**Executive Noodling:**
```python
# Session 1
"Create a React component with Redux"
[Generates component]
[Slow layer encodes: "React project, Redux state, functional components"]

# Session 2 (next day)
"Add another component"
[Slow layer provides: We use Redux, prefer functional components, tests in __tests__/]
[Generates consistent component without asking]
```

The slow layer maintains the "feel" of the project—architectural patterns, conventions, tech stack choices—that currently gets lost between sessions.

### 2. Surprise-Driven Code Review

**Surprise signals** when behavior deviates from expectations:

```python
# Noodling learns: "In this codebase, we always validate user input"

# User writes:
def process_user_data(data):
    return data.upper()  # No validation!

# High surprise → retrieves: similar functions that DO validate
# Suggests: "Should we add validation here? Similar functions use validate_input()"
```

This is the "something feels wrong" intuition experienced developers have.

### 3. Multi-Timescale Task Management

**Fast Layer** (seconds-minutes):
- Current function logic
- Immediate syntax decisions
- Variable naming in scope

**Medium Layer** (minutes-hours):
- Active task narrative ("implementing auth flow")
- Related code sections
- Recent refactoring decisions

**Slow Layer** (days-weeks):
- Overall architecture patterns
- Team coding standards
- Long-term technical debt awareness

This matches how human developers actually think about code at different timescales.

### 4. Episodic Code Memory

Instead of loading entire codebase:

```python
Current task: "Implement password reset"
                    ↓
Slow layer state triggers retrieval:
- "We solved email sending in user_registration.py"
- "Auth patterns in middleware/auth.js"
- "Similar form validation in profile_edit.py"
                    ↓
Build LLM context with ONLY relevant examples
```

**Advantages:**
- Efficient: Load what matters, not everything
- Scalable: Works with massive codebases
- Contextual: Retrieval guided by current task state

### 5. Learning Project-Specific Patterns

Over time, the Noodling learns:
- "This team prefers early returns over nested ifs"
- "Database queries always go through the Repository layer"
- "We write integration tests for API endpoints"

This isn't in the training data—it's learned from working with YOUR codebase.

## Implementation Path

### Phase 1: Proof of Concept (4-6 weeks)

**Goal**: Demonstrate hierarchical state helps with programming tasks

**Approach:**
1. Redefine phenomenal state dimensions:
   - Current: Affect vectors (valence, arousal, etc.)
   - New: Code state vectors (complexity, patterns, context depth, etc.)

2. Generate training data:
   - Programming sessions with temporal structure
   - Bug-fix scenarios (high surprise when code behaves unexpectedly)
   - Refactoring tasks (slow layer learns architectural patterns)

3. Metrics:
   - Context efficiency (tokens used vs. baseline)
   - Pattern recognition (learns project conventions)
   - Surprise accuracy (catches code smells)

**Deliverable**: Small programming agent that maintains state across a coding session better than stateless baseline

### Phase 2: Memory & Retrieval (6-8 weeks)

**Goal**: Effective episodic code memory system

**Approach:**
1. Code-aware embedding:
   - Store code snippets with semantic + structural info
   - AST-aware similarity matching
   - Context-dependent retrieval

2. Retrieval triggers:
   - Surprise-driven: "This looks wrong" → find correct examples
   - Task-driven: "Implementing X" → find related implementations
   - Pattern-driven: "Similar structure to Y" → architectural consistency

3. Integration:
   - Build LLM context from retrieved episodes
   - Balance between retrieved context and current state
   - Learn what retrieval patterns work

**Deliverable**: Selective memory that outperforms "load everything" approaches

### Phase 3: Long-Term Project Memory (8-10 weeks)

**Goal**: Slow layer that captures project "personality"

**Approach:**
1. Train slow layer on:
   - Multi-day coding sessions
   - Architecture decision documents
   - Code review patterns
   - Refactoring history

2. Evaluate:
   - Consistency with project conventions
   - Architectural decision recall
   - Long-term context maintenance

3. Optimize:
   - What information belongs in slow layer?
   - How quickly should it adapt?
   - Balance stability vs. flexibility

**Deliverable**: Agent that "knows" your codebase over weeks of work

### Phase 4: Production Integration (10-12 weeks)

**Goal**: Usable programming assistant

**Approach:**
1. IDE integration (VSCode extension)
2. Git-aware state management
3. Multi-user project understanding
4. Performance optimization
5. User studies with real developers

**Deliverable**: Beta product for early adopters

## Technical Challenges & Solutions

### Challenge 1: Phenomenal State Representation

**Problem**: Current 40-D state encodes affect, not code understanding

**Solution**:
- Redesign state space for code features:
  - Complexity indicators (nesting depth, cyclomatic complexity)
  - Pattern markers (architectural layers, design patterns in use)
  - Context depth (scope level, call stack awareness)
  - Quality signals (test coverage, type safety)
- Keep dimensionality low (40-100D) for efficiency
- Learn mapping through self-supervised training on code

### Challenge 2: Surprise in Code Context

**Problem**: What does "surprise" mean for code?

**Solution**:
- **Behavioral surprise**: Code behaves differently than predicted
- **Stylistic surprise**: Deviates from project patterns
- **Architectural surprise**: Violates expected layering/boundaries
- **Bug-pattern surprise**: Matches known anti-patterns

Train predictor on:
- Code execution traces (behavioral)
- Style-consistent codebases (stylistic)
- Well-architected projects (architectural)
- Buggy vs. fixed code pairs (bug patterns)

### Challenge 3: Retrieval at Scale

**Problem**: Large codebases = millions of potential retrievals

**Solution**:
- Hierarchical indexing:
  - Fast: Current file context
  - Medium: Related modules/packages
  - Slow: Cross-cutting architectural patterns
- Hybrid retrieval:
  - Vector similarity (semantic)
  - Graph search (structural dependencies)
  - Recency weighting (recent changes matter more)
- Lazy loading: Retrieve coarse first, refine as needed

### Challenge 4: Multi-Developer Projects

**Problem**: Multiple people working simultaneously

**Solution**:
- Shared slow layer (project conventions)
- Per-developer medium layer (personal task context)
- Per-session fast layer (immediate work)
- Conflict resolution when conventions diverge
- Social memory patterns from affective Noodlings apply here

## Business Value: Product Team Perspective

### Current Pain Points We Solve

1. **Context Window Costs**
   - Problem: Developers repeatedly load entire files/projects
   - Solution: Selective retrieval = 60-80% reduction in tokens
   - Impact: Lower API costs, faster responses

2. **Session Continuity**
   - Problem: "What were we working on yesterday?"
   - Solution: Persistent state across sessions
   - Impact: Faster task resumption, less cognitive overhead

3. **Onboarding Time**
   - Problem: New developers struggle with project conventions
   - Solution: Slow layer encodes implicit team knowledge
   - Impact: Faster ramp-up, more consistent code

4. **Code Review Efficiency**
   - Problem: Manual review catches style/pattern issues
   - Solution: Surprise signals + learned patterns
   - Impact: Catch issues before human review

5. **Technical Debt Management**
   - Problem: Easy to deviate from architecture over time
   - Solution: Slow layer maintains architectural awareness
   - Impact: More maintainable codebases

### Competitive Advantages

**vs. GitHub Copilot:**
- Stateless → Copilot suggests line-by-line
- Stateful → We maintain project understanding

**vs. ChatGPT/Claude Code:**
- Context window → Eventually hits limits, expensive
- Selective memory → Scales to any project size

**vs. Cursor/Replit:**
- Plan mode = prompt engineering
- Temporal hierarchy = architectural memory

**Novel Capabilities:**
- Cross-session learning (gets better over time with your project)
- Surprise-driven insights ("this looks wrong because...")
- Multi-timescale reasoning (immediate fix vs. architectural concern)
- Vibe programming (intuitive consistency, not just rule-following)

### Market Positioning

**Phase 1**: Developer tool (IDE extension)
- Target: Individual developers on large codebases
- Value: Better context management, session continuity
- Pricing: Subscription per developer

**Phase 2**: Team tool (shared project memory)
- Target: Development teams (5-50 developers)
- Value: Onboarding, consistency, code quality
- Pricing: Team subscription + per-project

**Phase 3**: Enterprise (organizational knowledge)
- Target: Large engineering orgs (100+ developers)
- Value: Cross-team patterns, architectural governance
- Pricing: Enterprise license

## End User Benefits

### For Individual Developers

**"I just want it to remember what we're doing"**

Current experience:
```
Monday: "Let's build feature X with approach Y"
Tuesday: Opens new chat: "How should I build feature X?"
         AI: "There are several approaches..." [suggests Z, not Y]
```

With Executive Noodlings:
```
Monday: "Let's build feature X with approach Y"
[Slow layer encodes: project, approach Y chosen, reasoning]
Tuesday: Continues coding
         AI: "Continuing with approach Y from yesterday..."
         [Maintains context without re-explaining]
```

**"It should know our codebase style"**

Current: AI suggests patterns that don't match your conventions
With Noodlings: Learns "this team uses X pattern" and stays consistent

**"Show me only what's relevant"**

Current: Either too much context (slow, expensive) or too little (misses connections)
With Noodlings: Smart retrieval based on current task and learned patterns

### For Teams

**"New hires should ramp up faster"**

Noodling as onboarding buddy:
- "Here's how we structure components in this project"
- "We always add tests in this specific way"
- "This pattern solves X, we use it for Y"

Encodes implicit team knowledge that's usually in senior developers' heads.

**"Maintain code quality at scale"**

Surprise mechanism catches:
- Style inconsistencies
- Architectural violations
- Missing tests/error handling
- Anti-patterns

Before they reach code review.

**"Don't repeat past mistakes"**

Episodic memory of:
- "We tried that approach, it caused performance issues"
- "This bug pattern has appeared before in module X"
- "That refactoring broke tests last time"

### For Organizations

**"Preserve knowledge when people leave"**

Slow layer captures:
- Architectural decisions and reasoning
- Project-specific patterns and conventions
- Lessons learned from past issues

Doesn't leave with the developer.

**"Scale best practices across teams"**

Learn patterns from high-performing teams:
- How they structure code
- Their testing strategies
- Architectural choices that worked

Apply to other projects.

**"Manage technical debt systematically"**

Track:
- Where architecture is being violated
- What shortcuts are accumulating
- Which areas need refactoring most

With temporal context: "This has been deviating for 3 weeks..."

## Research Value

Beyond product, this demonstrates:

1. **Affective architectures generalize**: Principles from social cognition apply to executive function

2. **Temporal hierarchy matters**: Different timescales need different processing, not just attention mechanisms

3. **Situated learning works**: Agents that learn from experience with specific contexts outperform general models

4. **Memory isn't just storage**: Selective, state-driven retrieval beats "remember everything"

5. **Surprise is general**: Predictive processing applies beyond perception

Publications:
- "Temporal Hierarchies for Code Understanding"
- "Surprise-Driven Programming Assistance"
- "From Affective to Executive: Generalizing Cognitive Architectures"

## Next Steps

### Immediate (Week 1-2)
1. Design executive state space (what dimensions encode code context?)
2. Create synthetic training data (programming sessions with temporal structure)
3. Prototype surprise metric for code (what makes code "feel wrong"?)

### Short Term (Month 1-3)
1. Train proof-of-concept executive Noodling
2. Evaluate vs. stateless baseline on programming tasks
3. Implement basic code memory retrieval
4. User studies with 5-10 developers

### Medium Term (Month 4-6)
1. Scale to real codebases
2. Build IDE integration (VSCode)
3. Refine slow layer learning (project conventions)
4. Beta with 50-100 developers

### Long Term (Month 7-12)
1. Production deployment
2. Team features (shared project memory)
3. Enterprise features (cross-project patterns)
4. Publish research findings

## Open Questions

1. **What should executive phenomenal state encode?**
   - Code complexity? Architectural layers? Test coverage?
   - How many dimensions? (Current: 40-D for affect)

2. **How does surprise work for code?**
   - Behavioral (execution traces)?
   - Stylistic (pattern matching)?
   - Both?

3. **What belongs in slow vs. medium vs. fast layers?**
   - Slow: Architecture, conventions (evolves over days/weeks)
   - Medium: Current task narrative (evolves over hours)
   - Fast: Immediate context (evolves second-to-second)

4. **How do we evaluate success?**
   - Context efficiency (tokens saved)?
   - Developer satisfaction?
   - Code quality metrics?
   - Time to task completion?

5. **Can we transfer learn from affective Noodlings?**
   - Does the architecture itself transfer?
   - Do we retrain from scratch?

6. **Multi-modal programming?**
   - Code + docs + git history + issue tracking?
   - How do different modalities inform state?

## Conclusion

Executive Noodlings apply principles from affective social agents to programming tasks. The core innovation isn't "add more context" but rather **maintain stateful understanding across timescales** and **selectively retrieve what matters**.

This is "vibe programming" made systematic: the AI develops an intuitive sense of your codebase that persists, adapts, and guides its suggestions. Not replacing human intuition, but augmenting it with machine memory and pattern recognition.

The architecture exists. The principles are validated (in social/affective domain). The question is: do these cognitive patterns generalize to executive function?

We think yes. And if so, this could be a fundamentally better way to build programming assistants.

---

## Appendix: Comparison Table

| Feature | Current AI Tools | Executive Noodlings |
|---------|-----------------|---------------------|
| **State** | Stateless | Persistent across sessions |
| **Context** | Full load or nothing | Selective retrieval |
| **Learning** | Pre-trained only | Adapts to your codebase |
| **Memory** | Conversation history | Episodic + temporal hierarchy |
| **Scale** | Limited by context window | Unlimited (selective retrieval) |
| **Consistency** | Varies between sessions | Maintains project understanding |
| **Surprise** | None | Flags unexpected patterns |
| **Cost** | High (repeated full context) | Lower (selective loading) |
| **Timescales** | Single (immediate response) | Three (fast/medium/slow) |
| **Team use** | Individual chats | Shared project memory |

## Appendix: Example Scenarios

### Scenario 1: Bug Fix with Historical Context

**Developer**: "This function is throwing errors in production"

**Current AI**:
```
"Let me see the function code..."
[Analyzes code in isolation]
"The error is likely from X, try Y"
```

**Executive Noodling**:
```
[Surprise signal: "This function's behavior differs from pattern"]
[Retrieves: Similar function that had the same bug 2 months ago]
"This looks like the issue we had in user_profile.py.
The problem was Z, we fixed it by doing W.
Should we apply the same solution here?"
```

### Scenario 2: New Feature Implementation

**Developer**: "Add OAuth login"

**Current AI**:
```
"Here's a basic OAuth implementation..."
[Generic solution, doesn't match project structure]
```

**Executive Noodling**:
```
[Slow layer: "This project uses middleware pattern for auth"]
[Retrieves: Existing auth middleware, session management]
"I'll add OAuth as a new auth middleware, similar to
our existing JWT implementation. We'll need to:
1. Add OAuth middleware (follows our existing pattern)
2. Update auth routes (consistent with current structure)
3. Add integration tests (like we do for other auth methods)
This maintains consistency with how we handle auth."
```

### Scenario 3: Refactoring Decision

**Developer**: "Should we refactor this module?"

**Current AI**:
```
"Refactoring could improve readability..."
[Generic advice]
```

**Executive Noodling**:
```
[Slow layer: "Major refactor 3 weeks ago caused test failures"]
[Medium layer: "Team is in feature freeze for release"]
[Fast layer: "Current module complexity = medium"]
"The code could benefit from refactoring, but:
- We had issues with our last big refactor
- We're in feature freeze (release in 1 week)
- Current complexity is manageable
Recommend: Wait until after release, then refactor as dedicated task.
I can create a tech debt ticket with specific improvement areas."
```
