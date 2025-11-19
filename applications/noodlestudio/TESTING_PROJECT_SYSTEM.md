# Testing the Project System + Ethics Council Import

## Step 1: Launch NoodleStudio

```bash
cd /Users/thistlequell/git/noodlings_clean/applications/noodlestudio
./launch.sh
```

## Step 2: Create a New Project

1. Click **File > New Project...**
2. Enter project name: `EthicsCouncil`
3. Choose location (suggest `~/Documents`)
4. Click OK

You should see:
- Window title updates to "NoodleSTUDIO - EthicsCouncil"
- Status bar shows "Created project: EthicsCouncil"
- Assets panel shows empty categories (Noodlings, Ensembles, etc.)

## Step 3: Verify Project Structure

Open terminal and check:

```bash
cd ~/Documents/EthicsCouncil
tree -L 2
```

Should see:
```
EthicsCouncil/
├── Assets/
│   ├── Ensembles/
│   ├── Noodlings/
│   ├── Prims/
│   ├── Scripts/
│   └── Stages/
├── Library/
├── Temp/
├── .gitignore
└── project.noodleproj
```

## Step 4: Import the Ethics Council Ensemble

We need to add an import feature! Let me add that now...

Actually, let's manually copy the ensemble for now:

```bash
cp /Users/thistlequell/git/noodlings_clean/applications/cmush/ensembles/ethics_council.ensemble \
   ~/Documents/EthicsCouncil/Assets/Ensembles/
```

Then in NoodleStudio, click the refresh button or reopen the project.

## Step 5: View the Ensemble

1. In Assets panel, click on "Assets" tab (if not already there)
2. Expand "Ensembles"
3. Right-click on "Ethics Council"
4. Select "View Details..."

You should see:
- Name: Ethics Council
- Description
- List of 11 agents:
  - Fred Rogers
  - Jim Henson
  - Werner Herzog
  - Shari Lewis
  - Kermit
  - Bjork
  - Nadya (Pussy Riot)
  - Emma Vigeland
  - Sam Seder
  - Brenda Laurel
  - Tim Schafer

## Step 6: Test File > Open Project

1. Close NoodleStudio
2. Relaunch
3. Click **File > Open Project...**
4. Navigate to ~/Documents/EthicsCouncil
5. Open it

Should reload the project with the ensemble visible.

## Next Steps

We need to implement:
1. **File > Import > Ensemble...** - Easy way to import ensembles
2. **File > Import > Noodling...** - Import individual Noodling recipes
3. **Load Ensemble to Stage** - Actually spawn the agents in noodleMUSH
4. **Connect NoodleStudio to noodleMUSH** - Bridge the IDE and the running world

But first - let's test what we have!
