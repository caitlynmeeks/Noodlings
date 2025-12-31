/**
 * Neural Interface - Yuki's Cybernetic Hacking Ability
 *
 * When Yuki interfaces with a computer system, she doesn't just access it -
 * she communes with the kami within. This script handles the mystical/technical
 * hybrid experience of her neural jack.
 *
 * Author: Caitlyn + Claude
 * Date: December 21, 2025
 */

// Exposed to Inspector
/** @type {number} */
var interfaceSpeed = 1.0;  // How fast she cracks systems (1.0 = normal)

/** @type {number} */
var mysticismLevel = 0.7;  // How much Shinto flavor in descriptions

/** @type {boolean} */
var verboseMode = true;    // Describe the experience in detail

// Private state
var isInterfacing = false;
var currentTarget = null;
var connectionStartTime = 0;

/**
 * Called when Yuki attempts to interface with a system
 */
function onInterface(target) {
    if (isInterfacing) {
        return {
            success: false,
            message: "*ears flatten* One is already communing with another kami. Patience."
        };
    }

    isInterfacing = true;
    currentTarget = target;
    connectionStartTime = context.noodle.time.time;

    // Emit the physical action
    context.noodle.events.emit("yuki_interface_start", {
        target: target,
        timestamp: connectionStartTime
    });

    // Generate the connection narrative
    var narrative = generateConnectionNarrative(target);

    return {
        success: true,
        message: narrative,
        duration: calculateDuration(target)
    };
}

/**
 * Generate mystical/technical description of interfacing
 */
function generateConnectionNarrative(target) {
    var parts = [];

    // Physical action
    parts.push("*trots to " + target.name + ", sniffs at the ports*");

    // Shinto perception (based on mysticismLevel)
    if (Math.random() < mysticismLevel) {
        var kamiDescriptions = [
            "The kami within this machine hum with... anticipation.",
            "An old spirit dwells here. It remembers better days.",
            "Young kami, this one. Eager. Untested.",
            "The silicon spirits whisper of data flows and forgotten passwords.",
            "Ah... this kami has been wounded. Someone was careless with it."
        ];
        parts.push(kamiDescriptions[Math.floor(Math.random() * kamiDescriptions.length)]);
    }

    // The connection
    parts.push("*sits on haunches, tail curling around paws*");
    parts.push("One moment.");
    parts.push("*extends data jack from behind ear, interfaces*");

    // The experience
    if (verboseMode) {
        var experiences = [
            "Ah— *yip!* —there. The firewalls part like shoji screens.",
            "*soft gasp* The data flows... like mountain streams after snow melt.",
            "*low pleased growl* This one knows these patterns. Old code, familiar.",
            "*ears swivel* So many voices in here... logs, processes, ghosts of users past.",
            "*eyes flicker with data* The encryption... beautiful. Like origami. *fox-laugh* But this old fox knows how to unfold paper."
        ];
        parts.push(experiences[Math.floor(Math.random() * experiences.length)]);
    }

    return parts.join(" ");
}

/**
 * Calculate how long the interface takes based on target complexity
 */
function calculateDuration(target) {
    var baseDuration = 3.0;  // seconds

    // Adjust for target complexity
    if (target.security) {
        baseDuration += target.security * 2.0;
    }

    // Adjust for Yuki's speed
    baseDuration /= interfaceSpeed;

    return baseDuration;
}

/**
 * Called when interface completes
 */
function onInterfaceComplete() {
    if (!isInterfacing) return;

    var duration = context.noodle.time.time - connectionStartTime;

    // Generate completion narrative
    var narrative = generateCompletionNarrative(duration);

    // Emit completion event
    context.noodle.events.emit("yuki_interface_complete", {
        target: currentTarget,
        duration: duration,
        success: true
    });

    // Clean up
    isInterfacing = false;
    currentTarget = null;

    return narrative;
}

/**
 * Generate narrative for successful completion
 */
function generateCompletionNarrative(duration) {
    var parts = [];

    if (duration < 2.0) {
        parts.push("*retracts data jack smoothly*");
        parts.push("Child's play. *fox-laugh*");
    } else if (duration < 5.0) {
        parts.push("*low satisfied growl*");
        parts.push("The kami and this one have reached... understanding.");
        parts.push("*retracts data jack*");
    } else {
        parts.push("*pants softly*");
        parts.push("A worthy challenge. The old spirits do not yield easily.");
        parts.push("*retracts data jack, shakes head*");
        parts.push("But eight centuries of patience... *tail swishes* ...outlasts any firewall.");
    }

    // Add wisdom coda
    if (Math.random() < mysticismLevel) {
        var wisdoms = [
            "The old ways and new ways dance together, yes?",
            "Even silicon dreams, if you know how to listen.",
            "Technology and nature... not so different, in the end.",
            "The kami remember everything. One need only ask politely."
        ];
        parts.push(wisdoms[Math.floor(Math.random() * wisdoms.length)]);
    }

    return parts.join(" ");
}

/**
 * Called when interface is interrupted
 */
function onInterfaceInterrupt(reason) {
    if (!isInterfacing) return;

    context.noodle.events.emit("yuki_interface_interrupted", {
        target: currentTarget,
        reason: reason
    });

    isInterfacing = false;
    currentTarget = null;

    return "*yip!* *retracts data jack sharply, ears flattening* " +
           "The connection... severed. *low growl* Most... unpleasant.";
}

/**
 * Check if currently interfacing
 */
function isCurrentlyInterfacing() {
    return isInterfacing;
}

/**
 * Get current interface target
 */
function getCurrentTarget() {
    return currentTarget;
}

// Export for facet system
module.exports = {
    onInterface: onInterface,
    onInterfaceComplete: onInterfaceComplete,
    onInterfaceInterrupt: onInterfaceInterrupt,
    isCurrentlyInterfacing: isCurrentlyInterfacing,
    getCurrentTarget: getCurrentTarget
};
