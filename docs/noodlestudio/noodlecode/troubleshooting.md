# NoodleCODE Troubleshooting Guide

## UI Issues

### Component not appearing after drag
**Symptom:** Dragged component from palette but nothing appears
**Solutions:**
1. Check you're in UI Canvas edit mode (component should be selected in Inspector)
2. Try dragging to a different position
3. Check if component landed outside visible canvas area - zoom out
4. Undo (Cmd+Z) and try again

### Can't select component in canvas
**Symptom:** Clicking on component doesn't select it
**Solutions:**
1. Component might be behind another - check hierarchy in Inspector
2. Component might be locked - check Inspector for lock icon
3. Try selecting from hierarchy instead of canvas

### Event wiring not showing
**Symptom:** Events section missing in Inspector
**Solutions:**
1. Make sure component (not Thing) is selected
2. Some components don't have events (Label, Panel)
3. Scroll down in Inspector - section might be collapsed

### Component properties not saving
**Symptom:** Changes revert when clicking away
**Solutions:**
1. Press Enter after typing in text fields
2. Click outside the field to confirm change
3. Check Console for save errors
4. Try Cmd+S to force save

---

## Facet Assembly Issues

### Assembly not executing
**Symptom:** Wired UI event to assembly but nothing happens
**Solutions:**
1. Check assembly is saved
2. Check wiring in UI Canvas Inspector - is assembly selected?
3. Check Console for errors
4. If continuous assembly, check `[Run in cognition loop]` is checked
5. Check Cognitive Cycles panel - is it paused?

### LLM not responding
**Symptom:** LLMFacet node shows no output
**Solutions:**
1. Check Console for API errors
2. Verify API key is configured (Preferences > API Keys)
3. Check network connection
4. Check if rate limited - wait and retry
5. Try reducing max_tokens

### Assembly stuck in phase
**Symptom:** Cognitive Cycles shows assembly stuck on one phase
**Solutions:**
1. Might be waiting for LLM response - check network
2. Click Step button in Cognitive Cycles to advance
3. Check for infinite loop in ScriptedFacet
4. Stop and restart preview (Cmd+. then Cmd+R)

### Facets not connecting
**Symptom:** Can't draw wire between facet ports
**Solutions:**
1. Check port types are compatible (hover to see type)
2. Output ports connect to input ports (not output-to-output)
3. Try connecting from output first, then to input
4. Delete existing connection first if changing

### Assembly input/output bindings not working
**Symptom:** UI values not reaching assembly or output not updating UI
**Solutions:**
1. Check binding syntax: `{component.property}`
2. Verify component name matches exactly
3. Check output binding targets a valid property
4. Look for binding errors in Console

---

## Build Issues

### Build fails with error
**Symptom:** Build dialog shows error
**Solutions:**
1. Read error message carefully
2. Common: missing required asset - check all referenced files exist
3. Common: invalid character in app name - use alphanumeric only
4. Check Console for detailed error
5. Try saving project first, then build

### Built app won't launch
**Symptom:** Double-click .app but nothing happens
**Solutions:**
1. macOS security: Right-click > Open first time
2. Check Console.app for crash logs
3. Rebuild with "Include Debug Info" checked
4. Test in preview mode first (Cmd+R) to ensure it works

### Built app missing features
**Symptom:** App works in preview but not after build
**Solutions:**
1. Check all assets are in project folder (not external references)
2. Verify all assemblies are saved
3. Check build log for skipped files
4. Ensure LLM provider configured for runtime use

---

## Preview Issues

### Preview won't start
**Symptom:** Cmd+R does nothing or errors
**Solutions:**
1. Check Console for errors
2. Save project first (Cmd+S)
3. Check for syntax errors in any ScriptedFacets
4. Try closing and reopening project

### Preview UI looks wrong
**Symptom:** UI layout different in preview vs editor
**Solutions:**
1. Check anchor settings on components
2. Preview might be different window size - test resize
3. Some styling only applies at runtime - this is expected

### Preview crashes immediately
**Symptom:** Preview opens then closes
**Solutions:**
1. Check Console for crash message
2. Look for infinite loops in continuous assemblies
3. Check for missing required assets
4. Try disabling continuous assemblies temporarily

---

## Connection Issues

### Can't connect to MCP server
**Symptom:** MCP facets show connection error
**Solutions:**
1. Check MCP server is running
2. Verify server URL in MCP Settings
3. Check network/firewall
4. Test with simpler MCP call first

### LLM provider not available
**Symptom:** "Provider not configured" error
**Solutions:**
1. Open Model Manager panel
2. Add provider with API key
3. Assign model to "Large" or "Noodle Code" label
4. Test with Provider dropdown in LLMFacet

---

## Cognitive Cycles Issues

### Thing not appearing in Cognitive Cycles
**Symptom:** Thing has assemblies but doesn't show in panel
**Solutions:**
1. Assembly must have `run_in_cognition_loop` checked
2. Preview must be running (Cmd+R)
3. Try Refresh button in panel
4. Check assembly has at least one facet

### Assembly shows "Error" status
**Symptom:** Red status in Cognitive Cycles
**Solutions:**
1. Click on assembly row to see error details
2. Check Console for full stack trace
3. Common: LLM timeout - retry
4. Common: Script error in ScriptedFacet - check syntax

---

## General Recovery

### Something is broken and I don't know why
1. **Save work:** Cmd+S (if possible)
2. **Check Console:** Look for red error messages
3. **Undo recent changes:** Cmd+Z multiple times
4. **Restart preview:** Cmd+. then Cmd+R
5. **Restart NoodleStudio:** Close and reopen application
6. **Check Cognitive Cycles:** Pause all, then step through

### Project won't open
**Symptom:** Project loading fails or hangs
**Solutions:**
1. Try recent backup in project folder
2. Check disk space
3. Open different project to verify app works
4. Check project folder permissions

### NoodleStudio unresponsive
**Symptom:** UI frozen, not responding to clicks
**Solutions:**
1. Wait 30 seconds (might be processing)
2. Check Activity Monitor for CPU usage
3. Try Cmd+. to interrupt
4. Force quit only as last resort (unsaved work lost)

---

## How to Report Issues to User

When something fails, tell the user:
1. What you were trying to do
2. What error appeared (exact message)
3. What you tried to fix it
4. Ask if they want to try a different approach

**Example:**
```
I tried to add a VisionFacet to the assembly but got error:
"VisionFacet requires vision-capable model"

I checked Model Manager and no vision model is configured.

Would you like me to:
1. Use a different approach without vision
2. Wait while you configure a vision model
3. Proceed with a placeholder that you can fix later
```
