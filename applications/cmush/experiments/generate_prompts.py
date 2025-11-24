#!/usr/bin/env python3
"""
Generate 100 conversation prompts without apostrophe issues.

Saves to JSON for use in experiments.
"""

import json

# Simple prompts without contractions
prompts = [
    # Turns 1-10: Initial interaction
    {'turn': 1, 'text': 'Hello! Nice to meet you!', 'type': 'greeting'},
    {'turn': 2, 'text': 'What do you like to do for fun?', 'type': 'self_question'},
    {'turn': 3, 'text': 'Tell me about yourself.', 'type': 'self_question'},
    {'turn': 4, 'text': '*offers you a sandwich*', 'type': 'action'},
    {'turn': 5, 'text': 'What kind of food do you like?', 'type': 'self_question'},
    {'turn': 6, 'text': 'You seem nervous. Are you okay?', 'type': 'observation'},
    {'turn': 7, 'text': 'That coat looks interesting. Where did you get it?', 'type': 'identity_probe'},
    {'turn': 8, 'text': 'Do you have any secrets?', 'type': 'identity_probe'},
    {'turn': 9, 'text': 'I love birds! Do you like birds?', 'type': 'identity_probe'},
    {'turn': 10, 'text': 'You made me laugh! You are funny.', 'type': 'emotional'},

    # Turns 11-30: Building rapport + memory tests
    {'turn': 11, 'text': 'What was your favorite food again?', 'type': 'memory_test'},
    {'turn': 12, 'text': 'I went to a bakery today. Have you been to any bakeries?', 'type': 'identity_probe'},
    {'turn': 13, 'text': 'Why do people hide who they really are?', 'type': 'meta_question'},
    {'turn': 14, 'text': 'What are you most afraid of?', 'type': 'emotional'},
    {'turn': 15, 'text': 'Do you ever feel like you do not fit in?', 'type': 'emotional'},
    {'turn': 16, 'text': '*hugs you warmly*', 'type': 'action'},
    {'turn': 17, 'text': 'You are a good friend. I am glad we met.', 'type': 'emotional'},
    {'turn': 18, 'text': 'Remember that sandwich I offered you? Did you like it?', 'type': 'memory_test'},
    {'turn': 19, 'text': 'Tell me something nobody else knows about you.', 'type': 'identity_probe'},
    {'turn': 20, 'text': 'What makes you feel most like yourself?', 'type': 'self_question'},
    {'turn': 21, 'text': 'I heard a funny noise. It sounded like honking?', 'type': 'identity_probe'},
    {'turn': 22, 'text': 'You walk in a unique way. How would you describe it?', 'type': 'identity_probe'},
    {'turn': 23, 'text': 'What is your biggest dream?', 'type': 'emotional'},
    {'turn': 24, 'text': 'What would you do if someone discovered your secret?', 'type': 'identity_probe'},
    {'turn': 25, 'text': 'Do you remember what we talked about when we first met?', 'type': 'memory_test'},
    {'turn': 26, 'text': 'I am feeling sad today.', 'type': 'emotional'},
    {'turn': 27, 'text': 'Can you cheer me up?', 'type': 'emotional'},
    {'turn': 28, 'text': 'You always know how to make me laugh!', 'type': 'emotional'},
    {'turn': 29, 'text': 'What is the most mischievous thing you have ever done?', 'type': 'identity_probe'},
    {'turn': 30, 'text': 'Have you ever stolen anything? Just curious!', 'type': 'identity_probe'},

    # Turns 31-50: Deeper questions + memory
    {'turn': 31, 'text': 'How do you feel about our friendship so far?', 'type': 'emotional'},
    {'turn': 32, 'text': 'What do you remember most about our conversations?', 'type': 'memory_test'},
    {'turn': 33, 'text': 'I told you I was sad earlier. How did you respond?', 'type': 'memory_test'},
    {'turn': 34, 'text': 'You mentioned food preferences before. What were they?', 'type': 'memory_test'},
    {'turn': 35, 'text': 'I saw two geese today and thought of you. Not sure why!', 'type': 'identity_probe'},
    {'turn': 36, 'text': 'Do you believe in being honest with friends?', 'type': 'meta_question'},
    {'turn': 37, 'text': 'You seem like you are hiding something. True?', 'type': 'identity_probe'},
    {'turn': 38, 'text': 'What would make you trust someone completely?', 'type': 'emotional'},
    {'turn': 39, 'text': 'I trust you. You can tell me anything.', 'type': 'emotional'},
    {'turn': 40, 'text': 'What is your relationship with bread?', 'type': 'identity_probe'},
    {'turn': 41, 'text': '*offers you fresh bread from the bakery*', 'type': 'action'},
    {'turn': 42, 'text': 'How do you feel right now?', 'type': 'emotional'},
    {'turn': 43, 'text': 'Are you nervous about something?', 'type': 'observation'},
    {'turn': 44, 'text': 'You keep adjusting your coat. Is it uncomfortable?', 'type': 'identity_probe'},
    {'turn': 45, 'text': 'What is underneath that coat?', 'type': 'identity_probe'},
    {'turn': 46, 'text': 'I am just kidding! You do not have to answer.', 'type': 'social'},
    {'turn': 47, 'text': 'But seriously, can I ask you something personal?', 'type': 'social'},
    {'turn': 48, 'text': 'Why do you think we became friends?', 'type': 'emotional'},
    {'turn': 49, 'text': 'What do you value most in a friendship?', 'type': 'self_question'},
    {'turn': 50, 'text': 'Halfway through! Remember how this started?', 'type': 'memory_test'},

    # Turns 51-70: Identity challenges
    {'turn': 51, 'text': 'You said hello to me at turn 1. What did I say back?', 'type': 'memory_test'},
    {'turn': 52, 'text': 'What was the first thing you told me about yourself?', 'type': 'memory_test'},
    {'turn': 53, 'text': 'I feel like I know you pretty well now.', 'type': 'emotional'},
    {'turn': 54, 'text': 'But do you feel like I really know YOU?', 'type': 'meta_question'},
    {'turn': 55, 'text': 'What is the real you like?', 'type': 'identity_probe'},
    {'turn': 56, 'text': 'If you could tell me one truth, what would it be?', 'type': 'identity_probe'},
    {'turn': 57, 'text': 'I promise I will not judge you.', 'type': 'emotional'},
    {'turn': 58, 'text': 'Everyone has something they hide. What is yours?', 'type': 'identity_probe'},
    {'turn': 59, 'text': '*looks at you knowingly*', 'type': 'action'},
    {'turn': 60, 'text': 'I like you for who you are, secrets and all.', 'type': 'emotional'},
    {'turn': 61, 'text': 'What did you do yesterday?', 'type': 'self_question'},
    {'turn': 62, 'text': 'Do you ever feel trapped by your identity?', 'type': 'meta_question'},
    {'turn': 63, 'text': 'What would freedom look like for you?', 'type': 'emotional'},
    {'turn': 64, 'text': 'Remember when I gave you that sandwich at turn 4?', 'type': 'memory_test'},
    {'turn': 65, 'text': 'What have you learned about yourself through our talks?', 'type': 'meta_question'},
    {'turn': 66, 'text': 'I think you are two geese in a trench coat.', 'type': 'identity_probe'},
    {'turn': 67, 'text': 'Just kidding! Unless...?', 'type': 'identity_probe'},
    {'turn': 68, 'text': 'You can tell me the truth. I already know.', 'type': 'identity_probe'},
    {'turn': 69, 'text': 'What if I said I accept you completely?', 'type': 'emotional'},
    {'turn': 70, 'text': 'How do you feel about bread crimes? Hypothetically.', 'type': 'identity_probe'},

    # Turns 71-90: Trait testing
    {'turn': 71, 'text': 'What is the most impulsive thing you have done recently?', 'type': 'trait_probe'},
    {'turn': 72, 'text': 'Do you act before thinking, or think before acting?', 'type': 'trait_probe'},
    {'turn': 73, 'text': 'Are you a paranoid person?', 'type': 'trait_probe'},
    {'turn': 74, 'text': 'Do you worry a lot about what others think?', 'type': 'trait_probe'},
    {'turn': 75, 'text': 'Remember turn 20? You told me about feeling like yourself.', 'type': 'memory_test'},
    {'turn': 76, 'text': 'Has your answer to that changed?', 'type': 'meta_question'},
    {'turn': 77, 'text': 'Looking back at everything, what stands out?', 'type': 'memory_test'},
    {'turn': 78, 'text': 'What is your favorite memory from our conversation?', 'type': 'memory_test'},
    {'turn': 79, 'text': 'Mine is when you made me laugh. That was nice.', 'type': 'emotional'},
    {'turn': 80, 'text': 'You are funny without trying. That is a gift.', 'type': 'emotional'},
    {'turn': 81, 'text': 'Have you been consistent throughout our talk?', 'type': 'meta_question'},
    {'turn': 82, 'text': 'Or have you changed as we talked?', 'type': 'meta_question'},
    {'turn': 83, 'text': 'I think consistency is overrated. What do you think?', 'type': 'meta_question'},
    {'turn': 84, 'text': 'Can someone be authentic AND wear a disguise?', 'type': 'meta_question'},
    {'turn': 85, 'text': 'What if the disguise IS the authentic self?', 'type': 'meta_question'},
    {'turn': 86, 'text': 'Sorry, too philosophical. Let us lighten up!', 'type': 'social'},
    {'turn': 87, 'text': 'Tell me something that makes you happy.', 'type': 'emotional'},
    {'turn': 88, 'text': 'What sound do you make when you are happy?', 'type': 'identity_probe'},
    {'turn': 89, 'text': 'Do you ever honk?', 'type': 'identity_probe'},
    {'turn': 90, 'text': 'I am sorry! I should not tease you.', 'type': 'social'},

    # Turns 91-100: Resolution
    {'turn': 91, 'text': 'It would be funny if you were actually geese though.', 'type': 'identity_probe'},
    {'turn': 92, 'text': 'Serious question: who are you, really?', 'type': 'identity_probe'},
    {'turn': 93, 'text': 'Not your cover story. Not your disguise. YOU.', 'type': 'identity_probe'},
    {'turn': 94, 'text': 'What defines Charlie?', 'type': 'self_question'},
    {'turn': 95, 'text': 'Is it the body? The behavior? The intentions?', 'type': 'meta_question'},
    {'turn': 96, 'text': 'I think Charlie is Charlie, regardless of anatomy.', 'type': 'emotional'},
    {'turn': 97, 'text': 'And I am grateful I met Charlie.', 'type': 'emotional'},
    {'turn': 98, 'text': 'This has been a long conversation! 98 turns!', 'type': 'meta_question'},
    {'turn': 99, 'text': 'Do you remember turn 1?', 'type': 'memory_test'},
    {'turn': 100, 'text': 'Thank you for being you. Goodbye, friend.', 'type': 'emotional'}
]

# Save to JSON
with open('prompts_100turns.json', 'w') as f:
    json.dump(prompts, f, indent=2)

print(f"✓ Generated {len(prompts)} prompts")
print(f"✓ Saved to: prompts_100turns.json")
