"""
Clickable Box Example - The Anklebiter Rezzer

WARNING: DO NOT CLICK

But of course everyone will click it...

Usage:
1. Create a prim called "Mysterious Box"
2. Attach this script (Component > Add Script)
3. Click the box in-world
4. Anklebiter rezzes!
5. Click again... another Anklebiter!
6. Click again... OH NO!

Author: Caitlyn + Claude
Date: November 18, 2025
"""

from noodlestudio.scripting import NoodleScript, Noodlings, Debug


class ClickableBox(NoodleScript):
    """
    A mysterious box that rezzes Anklebiters when clicked.

    Like a Minecraft mob spawner but WORSE!
    """

    def Start(self):
        """Initialize the box."""
        Debug.Log("Mysterious box initialized. (Don't click it!)")
        self.rez_count = 0
        self.max_rezzes = 10  # Safety limit!

    def OnClick(self, clicker):
        """
        Someone clicked the box!

        This is where the magic (chaos) happens.
        """
        Debug.Log(f"{clicker} clicked the mysterious box...")

        if self.rez_count >= self.max_rezzes:
            Debug.LogWarning("Box is exhausted! No more Anklebiters!")
            Noodlings.SendMessage(clicker, "The box rattles but nothing emerges. Thank goodness.")
            return

        # REZ AN ANKLEBITER!
        anklebiter = Noodlings.Rez(
            "anklebiter.noodling",
            position=self.transform.position,
            room=self.prim.room
        )

        if anklebiter:
            self.rez_count += 1

            # Messages based on how many have been rezzed
            if self.rez_count == 1:
                Debug.Log("Uh oh. You released an Anklebiter.")
                Noodlings.SendMessage(clicker, "A small, vicious creature emerges from the box and scurries around your ankles!")

            elif self.rez_count == 3:
                Debug.LogWarning("This is getting out of hand.")
                Noodlings.SendMessage(clicker, "Three Anklebiters now circle you menacingly. Maybe stop clicking?")

            elif self.rez_count == 5:
                Debug.LogError("WHY DO YOU KEEP CLICKING?!")
                Noodlings.SendMessage(clicker, "The Anklebiters are forming a PACK. This was a terrible idea.")

            elif self.rez_count >= self.max_rezzes:
                Debug.Log("Box is now empty. Disaster averted... barely.")
                Noodlings.SendMessage(clicker, "The box crumbles to dust. You monster.")
                # Destroy the box
                self.Destroy(delay=2.0)

            else:
                Debug.Log(f"Anklebiter #{self.rez_count} rezzed.")

    def OnUse(self, user):
        """Someone tried to use the box."""
        Debug.Log(f"{user} tried to use the box (nice try!)")
        Noodlings.SendMessage(user, "The box has no obvious latch or keyhole. There's just a big red button that says 'DO NOT PRESS'.")
