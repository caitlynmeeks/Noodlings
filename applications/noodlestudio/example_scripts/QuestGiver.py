"""
Quest Giver Example - NPC with a quest

A mysterious hooded figure who gives you a quest when you talk to them.

Usage:
1. Spawn a Noodling (e.g., "Mysterious Stranger")
2. Attach this script
3. Talk to them in-world
4. They give you a quest!
5. Complete objectives to finish quest

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from noodlestudio.scripting import NoodleScript, Noodlings, Debug


class QuestGiver(NoodleScript):
    """NPC that gives quests when you talk to them."""

    def Start(self):
        """Initialize quest state."""
        self.quest_given = False
        self.quest_completed = False

        self.quest_name = "The Missing Tensor Taffy"
        self.quest_description = "Find the stolen tensor taffy and return it to me."

        self.objectives = [
            "Talk to the baker about the missing taffy",
            "Search the warehouse for clues",
            "Confront the thief",
            "Return the taffy"
        ]

        self.current_objective = 0

        Debug.Log(f"Quest giver initialized: {self.quest_name}")

    def OnHear(self, speaker, message):
        """
        Listen for specific trigger phrases.

        If someone says "quest" or "help", give them the quest!
        """
        message_lower = message.lower()

        if not self.quest_given and ('quest' in message_lower or 'help' in message_lower):
            # Give the quest!
            self.give_quest(speaker)

        elif self.quest_given and not self.quest_completed:
            # Check for quest items
            if 'taffy' in message_lower and 'have' in message_lower:
                self.complete_quest(speaker)

    def give_quest(self, player):
        """Give the quest to player."""
        self.quest_given = True

        Debug.Log(f"Giving quest to {player}: {self.quest_name}")

        Noodlings.SendMessage(
            player,
            f"Ah, you seek adventure? I have a task for you...\n\n"
            f"**{self.quest_name}**\n"
            f"{self.quest_description}\n\n"
            f"Current objective: {self.objectives[self.current_objective]}"
        )

    def complete_quest(self, player):
        """Player completed the quest!"""
        self.quest_completed = True

        Debug.Log(f"{player} completed quest: {self.quest_name}")

        Noodlings.SendMessage(
            player,
            "You've done it! The tensor taffy is safe! Here, take this reward..."
        )

        # Rez reward item
        reward = Noodlings.RezPrim("prop", "Golden Noodling Statue", room=self.prim.room)
        Debug.Log(f"Rezzed quest reward: {reward.id}")

    def OnClick(self, clicker):
        """Shortcut - clicking also gives quest."""
        if not self.quest_given:
            self.give_quest(clicker)
        elif self.quest_completed:
            Noodlings.SendMessage(clicker, "Thank you again, brave adventurer.")
        else:
            objective = self.objectives[self.current_objective]
            Noodlings.SendMessage(clicker, f"Your current objective: {objective}")
