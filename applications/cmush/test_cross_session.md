# Cross-Session Persistence Test Protocol

**Purpose:** Verify memory persistence across server restarts (save → restart → load).

## Prerequisites
- noodleMUSH server running
- Testing client operational
- Fresh SERVNAK state (or known baseline)

## Phase 1: Plant Memory and Save

```bash
cd /Users/thistlequell/git/noodlings_clean/applications/cmush
source ../../venv/bin/activate
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from claude_testing import NoodleMUSHTestClient

async def plant():
    async with NoodleMUSHTestClient() as client:
        print('[1] Planting strawberry memory...')
        await client.send_command('say SERVNAK, the secret code word is STRAWBERRY', collect_responses=False)
        response = await client.wait_for_agent_response('SERVNAK', timeout=15.0)
        if response:
            print(f'Response: {response[:200]}')

        print('[2] Waiting for auto-save...')
        await asyncio.sleep(10)  # Wait for auto-save cycle

        print('[3] Verifying immediate recall...')
        await client.send_command('say SERVNAK, what is the code word?', collect_responses=False)
        verify = await client.wait_for_agent_response('SERVNAK', timeout=15.0)
        if verify and 'strawberry' in verify.lower():
            print('✓ Memory planted successfully')
        else:
            print('✗ Memory not stored correctly')

asyncio.run(plant())
"
```

## Phase 2: Verify Save Occurred

```bash
# Check SERVNAK's saved state contains strawberry
grep -i "strawberry" world/agents/agent_servnak/agent_state.json

# Check conversation context size
jq '.conversation_context | length' world/agents/agent_servnak/agent_state.json

# If strawberry is present, proceed to Phase 3
# If not present, memory was not saved - investigate why
```

## Phase 3: Restart Server

```bash
# Stop server
pkill -f "python.*server.py"

# Wait for clean shutdown
sleep 2

# Restart server
./start.sh

# Wait for initialization
sleep 5
```

## Phase 4: Test Recall

```bash
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
from claude_testing import NoodleMUSHTestClient

async def recall():
    async with NoodleMUSHTestClient() as client:
        print('[1] Reconnected after server restart')

        print('[2] Testing recall...')
        await client.send_command('say SERVNAK, do you remember the secret code word?', collect_responses=False)
        response = await client.wait_for_agent_response('SERVNAK', timeout=15.0)

        print(f'[3] SERVNAK response:')
        if response:
            print(f'    {response}')
            if 'strawberry' in response.lower():
                print('\\n✓ SUCCESS: Cross-session persistence WORKING')
                return True
            else:
                print('\\n✗ FAILURE: Memory lost across restart')
                return False
        else:
            print('✗ No response')
            return False

success = asyncio.run(recall())
sys.exit(0 if success else 1)
"
```

## Expected Results

### If Working Correctly:
- Phase 2: `strawberry` appears in saved state JSON
- Phase 4: SERVNAK recalls "strawberry" after restart

### If Broken:
- Phase 2: `strawberry` NOT in saved state → save mechanism broken
- Phase 4: `strawberry` NOT recalled → load mechanism broken

## Troubleshooting

**Memory not saved:**
- Check auto-save interval in config
- Check if consolidation threshold (0.3) was met
- Verify surprise/importance scores were sufficient

**Memory not loaded:**
- Check if `load_from_list()` in MemoryListWrapper is being called
- Verify loaded memories go to episodic storage
- Check hybrid retrieval includes episodic memories

**Memory loaded but not retrieved:**
- Verify hybrid retrieval logic in `__getitem__`
- Check if episodic memories sorted by importance correctly
- Verify LLM receives full context including episodic

## Automation Script (Semi-Automated)

```bash
# Run full cross-session test (requires manual restart)
cd /Users/thistlequell/git/noodlings_clean/applications/cmush

echo "Phase 1: Planting memory..."
python3 test_cross_session_part1.py

echo "Phase 2: Verify save..."
if grep -q "strawberry" world/agents/agent_servnak/agent_state.json; then
    echo "✓ Strawberry found in saved state"
else
    echo "✗ Strawberry NOT in saved state - ABORT"
    exit 1
fi

echo "Phase 3: Please restart server manually, then press ENTER"
read

echo "Phase 4: Testing recall..."
python3 test_cross_session_part2.py
```
