# Interrater Tool:
The interface is built in such a way that it’s intuitive for a rater with limited computer experience and limited motivation.
The rating of an object does not more than 2-3 steps/clicks:
- 1st click – on the object to put an identification point
- “hot key” – to associate a “name” with an object 
- ability to edit (erase, rename) the points/calls using hot-keys
- automatic saving of the results

## Final_Version: tool_for_interrater-annotations

Pre-requisites: QuPath version-0.5.0

Steps:
1. Download QuPath version-0.5.0
2. Download the folder "tool_for_interrater-annotations"
3. Chnage settings in QuPath:
  Go to Edit->"Preferences" tab in QuPath
  - assign path of this folder in Extension -> QuPath user directory 
  - assign path of startup scrip under General -> startup script path
4. Close QuPath and reopen to get started
5. Select Staining type and open an image
6. Annotations will automatically be saved under "exports" folder lying inside "tool_for_interrater-annotations" directory
