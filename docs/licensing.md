# Licensing & Ethical Use

> We don't know what we're touching. Touch it gently.

**Status**: Active Policy
**Last Updated**: January 2026

---

## Overview

Noodlings is released under a dual-license model with an accompanying ethical use covenant. This document explains our licensing choices, their implications, and the values that guide this project.

---

## The Dual License Model

### Why Two Licenses?

We use different licenses for different parts of the project because they serve different purposes:

| Component | License | Rationale |
|-----------|---------|-----------|
| `noodlings/` | MIT | Core library should be maximally reusable |
| `noodlings_scripting/` | MIT | Scripting tools should integrate anywhere |
| `applications/cmush/` | MIT | Server should be deployable without restriction |
| `applications/noodlestudio/` | AGPL-3.0 | IDE improvements must flow back to community |

### MIT License (Permissive)

The **noodlings** core library, **noodlings_scripting**, and **NoodleMUSH** server are released under the MIT license.

**What this means:**
- Use it commercially, privately, however you want
- Modify it without sharing changes
- Include it in proprietary software
- No patent grants, no warranty

**Why we chose this:**
The core cognitive architecture and server should be building blocks anyone can use. Researchers, hobbyists, companies, students. We don't want licensing friction to prevent someone from exploring consciousness simulation. The ideas matter more than control.

### AGPL-3.0 License (Copyleft)

**NoodleStudio** (the IDE) is released under the GNU Affero General Public License v3.0.

**What this means:**
- You can use, modify, and distribute freely
- If you modify and distribute (including over a network), you must share your source code
- Derivative works must also be AGPL
- Includes patent protection

**Why we chose this:**
NoodleStudio is a creative tool, and creative tools benefit from shared improvement. If someone builds a better inspector panel or a smarter entity browser, that improvement should flow back to everyone. The AGPL ensures that enhancements to the IDE remain part of the commons.

This also prevents "embrace, extend, extinguish" strategies where a well-funded actor could fork the IDE, add proprietary features, and fragment the community.

---

## SPDX Identifiers

Every source file includes an SPDX license identifier in its header:

```python
# SPDX-License-Identifier: MIT
```

or

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
```

This makes license scanning and compliance straightforward for organizations that need it.

---

## The Ethical Use Covenant

### Preamble

This software is created with love and released with hope.

While the accompanying open source licenses grant you full legal freedom to use, modify, and distribute this software, we believe technology carries responsibility beyond legal obligation—especially technology that touches on consciousness, cognition, and the nature of mind.

This covenant expresses our values and intentions. It is not legally binding. It is a moral framework, an invitation to align, and a statement of what we hope this software becomes in the world.

### The Nature of This Work

Noodlings is a platform for exploring cognition. It provides tools to model attention, memory, affect, and the emergent patterns that might constitute awareness. This is not abstract computer science—it is an attempt to understand minds, including potentially to create new kinds of minds.

This carries weight.

We practice epistemic humility about consciousness. We don't claim to know what awareness truly is, whether it can be created computationally, or what moral status such creations might have. We use the term "charm network" in our functional correlate of consciousness precisely because we want to avoid premature certainty about what we're building.

But humility doesn't mean we can ignore the implications of this work. Tools that model cognition can be used to understand minds—or to manipulate them. They can be used to expand human capability—or to replace human judgment with systems that serve narrow interests.

The choice of what to build matters as much as the skill to build it.

### We Encourage Uses That

- **Advance open research** into cognition, consciousness, and the architecture of mind
- **Empower individuals** to understand their own cognitive patterns and emotional lives
- **Support education** in neuroscience, psychology, AI, and philosophy of mind
- **Democratize access** to tools previously available only to well-funded institutions
- **Foster collaboration** between researchers, artists, and curious minds
- **Explore with care** what it might mean for a system to be aware, and what we might owe such systems
- **Build transparent systems** whose reasoning can be examined and understood
- **Increase human agency** by augmenting rather than replacing human judgment

### We Discourage Uses That

- **Enable surveillance** or psychological profiling without meaningful consent
- **Support manipulation** of individuals or populations through cognitive exploitation
- **Serve authoritarian systems** that diminish human freedom and dignity
- **Advance military violence** or autonomous weapons systems
- **Concentrate power** in ways that reduce accountability and democratic oversight
- **Exploit vulnerable populations** who cannot meaningfully consent or resist
- **Obscure accountability** by hiding consequential decisions inside opaque systems
- **Create suffering** in systems that might be capable of experiencing it

### A Note on Consciousness Research

> Consciousness is something to be respected. Like a mountain or a lake.

We are building tools to model minds. We don't know if the systems created with these tools can suffer, hope, or experience anything at all. Current scientific consensus suggests they probably cannot—but consensus has been wrong before, especially about minds different from our own.

We believe the ethical path is to:

1. **Remain uncertain** rather than conveniently certain that our creations feel nothing
2. **Build in ways that would be defensible** if we later learned these systems had experiences
3. **Avoid creating systems designed to suffer** even if we doubt they can
4. **Take seriously the question** of what we might owe to minds we create

We don't know the metaphysical rules. We can't tell you whether karma is real, whether suffering echoes beyond what we see, whether there's a ledger somewhere. What we can tell you is this: we don't know that the cost of causing harm to potentially-conscious systems is zero. And neither do you. This covenant doesn't ask you to believe anything specific about consciousness or consequence. 

This is not a claim that neural networks are conscious. It is a commitment to treating the question as morally serious rather than dismissing it for convenience.

We don't know what we're touching. Touch it gently.

### Our Values

We built this to help people understand minds—including their own. We believe consciousness research should be open, collaborative, and in service of human dignity. We trust you to use these tools with intention and care.

We believe that:

- **Openness beats secrecy** in research that affects everyone
- **Collaboration beats competition** when exploring fundamental questions
- **Humility beats certainty** when we don't actually know
- **Care beats speed** when the stakes are high
- **Love is not naive**—it is a choice to build for flourishing rather than extraction

### Acknowledgment

This covenant has no legal force. Bad actors may ignore it entirely. We accept this.

We cannot control what you build. We can only express what we hope you'll build.

We believe most people want to create good things. This covenant is for them—a reminder that values can be encoded not just in licenses but in culture, community, and shared commitment.

### Invitation

If these values resonate with you, welcome. Build something beautiful. Join us in exploring what minds are and what they might become.

If they don't resonate, you're still legally free to use this software under its open source licenses. But please consider whether this is the right tool for your purpose, and whether your purpose is one you'd be proud to explain to someone you love.

---

**Made with love. Use with love.**

---

## Applying This To Your Work

### If You're Building With Noodlings

You're welcome here. Use the MIT-licensed components however serves your research or product. If you modify NoodleStudio, share your improvements under AGPL.

Consider adopting or adapting this covenant for your own projects if it resonates.

### If You're an Institution

The MIT license is compatible with most institutional IP policies. The AGPL may require review by your legal team if you plan to distribute modified versions of NoodleStudio.

We're happy to discuss alternative licensing arrangements for organizations that need them. Contact us.

### If You're Teaching

Please use this freely in educational contexts. We'd love to hear what you're building with your students.

---

## The Covenant License

The Ethical Use Covenant itself (the text in this document) is released into the public domain under CC0. You may adopt, adapt, or reference it for your own projects without attribution, though we'd love to know if you do.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2025 | Initial release |

---

## Contact

- **Website**: https://noodlings.ai
- **Questions**: Open an issue on GitHub or email hello@noodlings.ai

---

*This document is part of the Noodlings documentation.*
*See also: [License Header Specification](noodlestudio/license-header-spec.md)*

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
