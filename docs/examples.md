# Complete Examples

Real-world usage patterns for the Noodlings Scripting API.

## Example 1: Circadian Model Switching

Switch models based on time of day to save API costs while maintaining quality during active hours.

```javascript
function process(inputs, context) {
    var models = context.noodle.models;
    var hour = new Date().getHours();

    // Define time periods
    var isNight = (hour >= 22 || hour < 6);      // 10 PM - 6 AM
    var isPeakHours = (hour >= 9 && hour < 18);  // 9 AM - 6 PM

    if (isNight) {
        // Night mode: Local models only
        models.set_label("SMALL", "ollama", "deepseek-r1:7b");
        models.set_label("MEDIUM", "ollama", "deepseek-r1:14b");
        models.set_label("LARGE", "ollama", "deepseek-r1:70b");
        context.log("🌙 Night mode: Using local models");
    } else if (isPeakHours) {
        // Peak hours: Best quality cloud models
        models.set_label("SMALL", "anthropic", "claude-haiku-4.0");
        models.set_label("MEDIUM", "anthropic", "claude-sonnet-4.5");
        models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("☀️ Peak hours: Using Claude models");
    } else {
        // Off-peak: Mix of local and cloud
        models.set_label("SMALL", "ollama", "deepseek-r1:7b");
        models.set_label("MEDIUM", "anthropic", "claude-haiku-4.0");
        models.set_label("LARGE", "anthropic", "claude-sonnet-4.5");
        context.log("🌤️ Off-peak: Mixed models");
    }

    return {
        mode: isNight ? "night" : (isPeakHours ? "peak" : "off-peak"),
        hour: hour
    };
}
```

## Example 2: Task Complexity Analysis

Dynamically choose models based on input complexity to optimize cost/quality.

```javascript
function process(inputs, context) {
    var models = context.noodle.models;

    // Analyze input complexity
    var text = inputs.text || "";
    var wordCount = text.split(' ').length;
    var hasCode = text.includes("```") || text.includes("function");
    var hasMultipleQuestions = (text.match(/\?/g) || []).length > 2;

    // Calculate complexity score (0-1)
    var complexity = 0;
    complexity += Math.min(wordCount / 1000, 1.0) * 0.4;  // Length factor
    complexity += hasCode ? 0.3 : 0;                      // Code factor
    complexity += hasMultipleQuestions ? 0.3 : 0;         // Question factor

    context.log("Complexity score: " + complexity.toFixed(2));

    // Route to appropriate model
    if (complexity > 0.7) {
        models.set_label("LARGE", "anthropic", "claude-opus-4.5");
        context.log("High complexity → Claude Opus");
    } else if (complexity > 0.4) {
        models.set_label("LARGE", "anthropic", "claude-sonnet-4.5");
        context.log("Medium complexity → Claude Sonnet");
    } else {
        models.set_label("LARGE", "ollama", "deepseek-r1:70b");
        context.log("Low complexity → Local DeepSeek");
    }

    return {
        complexity: complexity,
        word_count: wordCount,
        has_code: hasCode
    };
}
```

## Example 3: Affect-Driven Temperature

Adjust LLM temperature based on agent's emotional state for more authentic responses.

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly(context.agent.id);
    var mind = assembly.get_facet_by_name("Red's Mind");

    if (!mind) {
        return {error: "Mind facet not found"};
    }

    // Get affect state from CharmNetwork
    var valence = inputs.affect_valence || 0.0;    // -1 to 1
    var arousal = inputs.affect_arousal || 0.5;    // 0 to 1
    var dominance = inputs.affect_dominance || 0.5; // 0 to 1

    // Calculate temperature based on affect
    // High arousal = more chaotic/creative
    // Positive valence = more exploratory
    // High dominance = more confident/assertive

    var base_temp = 0.7;

    // Arousal: 0.5 baseline → 0.0 to 1.0 adjustment
    var arousal_adj = (arousal - 0.5) * 0.6;

    // Valence: Positive = slight creativity boost
    var valence_adj = Math.max(0, valence) * 0.2;

    // Dominance: Low dominance = more conservative
    var dominance_adj = (dominance < 0.3) ? -0.2 : 0;

    var new_temp = base_temp + arousal_adj + valence_adj + dominance_adj;
    new_temp = Math.max(0.1, Math.min(new_temp, 1.5));  // Clamp to [0.1, 1.5]

    mind.set_property("temperature", new_temp);

    context.log("Affect-driven temperature adjustment:");
    context.log("  Valence: " + valence.toFixed(2) + " | Arousal: " + arousal.toFixed(2) + " | Dominance: " + dominance.toFixed(2));
    context.log("  Temperature: " + new_temp.toFixed(2));

    return {
        temperature: new_temp,
        valence: valence,
        arousal: arousal,
        dominance: dominance
    };
}
```

## Example 4: Procedural Neural Architecture

Generate neural topologies based on task requirements.

```javascript
function process(inputs, context) {
    var neural = context.noodle.neural;

    // Task parameters
    var sequenceLength = inputs.sequence_length || 100;
    var requires_long_memory = sequenceLength > 500;

    // Create network
    var network = neural.create_network("AdaptiveCharmNet");

    // Input node
    var input_id = network.create_node("Input", {
        output_dim: 64,
        position: [50, 200]
    });

    var prev_id = input_id;
    var x_pos = 200;

    // Fast timescale (always)
    var fast_lstm = network.create_node("LSTM", {
        hidden_dim: 16,
        position: [x_pos, 200]
    });
    network.connect(prev_id, "out", fast_lstm, "input");
    prev_id = fast_lstm;
    x_pos += 150;

    // Medium timescale (always)
    var medium_lstm = network.create_node("LSTM", {
        hidden_dim: 16,
        position: [x_pos, 200]
    });
    network.connect(prev_id, "out", medium_lstm, "input");
    prev_id = medium_lstm;
    x_pos += 150;

    // Add extra LSTM for long sequences
    if (requires_long_memory) {
        var extra_lstm = network.create_node("LSTM", {
            hidden_dim: 32,
            position: [x_pos, 200]
        });
        network.connect(prev_id, "out", extra_lstm, "input");
        prev_id = extra_lstm;
        x_pos += 150;
        context.log("Added extra LSTM layer for long sequences");
    }

    // Slow timescale (GRU)
    var slow_gru = network.create_node("GRU", {
        hidden_dim: 8,
        position: [x_pos, 200]
    });
    network.connect(prev_id, "out", slow_gru, "input");
    prev_id = slow_gru;
    x_pos += 150;

    // Affect head
    var affect_head = network.create_node("AffectHead", {
        output_dim: 5,
        position: [x_pos, 200]
    });
    network.connect(prev_id, "out", affect_head, "input");

    // Generate code and save
    var code = network.generate_mlx_code();
    var params = network.get_parameter_count();

    network.save("adaptive_charmnet_seq" + sequenceLength + ".nncanvas");

    context.log("Generated network:");
    context.log("  Parameters: " + params);
    context.log("  Code length: " + code.length + " chars");
    context.log("  Long memory: " + requires_long_memory);

    return {
        parameters: params,
        code_length: code.length,
        long_memory: requires_long_memory
    };
}
```

## Example 5: Self-Healing Assembly

Monitor facet health and automatically fix broken configurations.

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly(context.agent.id);
    var models = context.noodle.models;

    var issues_found = [];
    var fixes_applied = [];

    // Check all LLM facets
    var facets = assembly.list_facets();

    facets.forEach(function(f) {
        if (f.type === "LLMFacet") {
            var facet = assembly.get_facet(f.id);

            // Check 1: Model label assigned?
            var model = facet.get_property("model");
            if (!model) {
                issues_found.push(f.name + ": No model assigned");
                facet.set_property("model", "MEDIUM");
                fixes_applied.push(f.name + ": Set model to MEDIUM");
            } else {
                // Check 2: Is the model label valid?
                var assignment = models.get_label(model);
                if (!assignment || !assignment.provider) {
                    issues_found.push(f.name + ": Invalid model label '" + model + "'");
                    facet.set_property("model", "MEDIUM");
                    fixes_applied.push(f.name + ": Reassigned to MEDIUM");
                }
            }

            // Check 3: Temperature in valid range?
            var temp = facet.get_property("temperature");
            if (temp !== null && (temp < 0 || temp > 2.0)) {
                issues_found.push(f.name + ": Invalid temperature " + temp);
                facet.set_property("temperature", 0.7);
                fixes_applied.push(f.name + ": Reset temperature to 0.7");
            }

            // Check 4: Has prompt?
            var prompt = facet.get_property("prompt");
            if (!prompt || prompt.length < 10) {
                issues_found.push(f.name + ": Prompt too short or missing");
                facet.set_property("prompt", "You are a helpful assistant.");
                fixes_applied.push(f.name + ": Added default prompt");
            }
        }
    });

    // Report results
    if (issues_found.length > 0) {
        context.log("=== Self-Healing Report ===");
        context.log("Issues found: " + issues_found.length);
        issues_found.forEach(function(issue) {
            context.log("  ⚠️ " + issue);
        });
        context.log("");
        context.log("Fixes applied: " + fixes_applied.length);
        fixes_applied.forEach(function(fix) {
            context.log("  ✅ " + fix);
        });

        // Save healed assembly
        assembly.save("facet_assemblies/" + context.agent.id + "_healed.yaml");
    } else {
        context.log("✅ Assembly health check: All facets OK");
    }

    return {
        issues_found: issues_found.length,
        fixes_applied: fixes_applied.length,
        healthy: issues_found.length === 0
    };
}
```

## Example 6: Neural Topology Inspector

Analyze and visualize neural network structure.

```javascript
function process(inputs, context) {
    var neural = context.noodle.neural;

    // Load CharmNetwork
    var network = neural.load("facet_assemblies/charm_networks/default.nncanvas");
    if (!network) {
        return {error: "Could not load network"};
    }

    context.log("=== CharmNetwork Topology Analysis ===");
    context.log("");

    // Find all nodes by type
    var node_types = {};

    // Known node names in default topology
    var node_names = [
        "Fast_LSTM", "Medium_LSTM", "Slow_GRU",
        "State_Concat", "Affect_Head"
    ];

    node_names.forEach(function(name) {
        var node_id = network.get_node_by_name(name);
        if (node_id) {
            var node = network.get_node(node_id);
            if (node) {
                context.log("--- " + node.name + " ---");
                context.log("  Type: " + node.type);

                // Show key properties
                if (node.properties.hidden_dim) {
                    context.log("  Hidden dim: " + node.properties.hidden_dim);
                }
                if (node.properties.output_dim) {
                    context.log("  Output dim: " + node.properties.output_dim);
                }

                context.log("  Position: [" + node.position[0] + ", " + node.position[1] + "]");
                context.log("");

                // Count by type
                var type = node.type;
                node_types[type] = (node_types[type] || 0) + 1;
            }
        }
    });

    // Summary
    context.log("=== Summary ===");
    for (var type in node_types) {
        context.log("  " + type + ": " + node_types[type]);
    }

    var total_params = network.get_parameter_count();
    context.log("  Total parameters: " + total_params);

    // Generate code
    var code = network.generate_mlx_code();
    context.log("  Generated code: " + code.length + " chars");

    return {
        node_types: node_types,
        total_parameters: total_params,
        code_length: code.length
    };
}
```

## Example 7: Dynamic Prompt Injection

Modify LLM prompts based on conversation history and context.

```javascript
function process(inputs, context) {
    var assembly = context.noodle.agents.get_assembly(context.agent.id);
    var mind = assembly.get_facet_by_name("Red's Mind");

    if (!mind) {
        return {error: "Mind facet not found"};
    }

    // Get conversation history from storage
    var history = context.storage.conversation_history || [];

    // Analyze recent interactions
    var recentMessages = history.slice(-5);
    var hasQuestions = recentMessages.some(function(msg) {
        return msg.text.includes('?');
    });
    var hasCode = recentMessages.some(function(msg) {
        return msg.text.includes('```');
    });

    // Get base prompt
    var base_prompt = mind.get_property("prompt") || "";

    // Build dynamic additions
    var additions = [];

    if (hasQuestions) {
        additions.push("The user is asking questions. Be thorough and clear in explanations.");
    }

    if (hasCode) {
        additions.push("The conversation involves code. Focus on technical accuracy.");
    }

    // Time-based additions
    var hour = new Date().getHours();
    if (hour >= 22 || hour < 6) {
        additions.push("It's late at night. Be concise - the user might be tired.");
    }

    // Affect-based additions
    if (inputs.affect_valence < -0.3) {
        additions.push("The agent is feeling negative. Respond with empathy.");
    }

    // Combine
    var enhanced_prompt = base_prompt;
    if (additions.length > 0) {
        enhanced_prompt += "\n\nCurrent context:\n" + additions.join("\n");
    }

    // Apply
    mind.set_property("prompt", enhanced_prompt);

    context.log("Enhanced prompt with " + additions.length + " contextual additions");

    return {
        additions: additions.length,
        has_questions: hasQuestions,
        has_code: hasCode
    };
}
```

## See Also

- [Quick Start Guide](quick-start.md)
- [API Reference](api/overview.md)
- [Troubleshooting](troubleshooting.md)
