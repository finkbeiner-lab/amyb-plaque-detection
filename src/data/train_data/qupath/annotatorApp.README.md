QuPath Annotator Tool Script: src/data/train_data/qupath/annotatorApp.groovy
  The annotatorApp script may be executed once as a startup script (although there is no limit on the number of times; subsequent invocations will simply reinstantiate the same tool command). Upon completion, the script's final line installs the tool's Runnable as a native QuPath command, and returns a handle to the menu item which it corresponds to. This menu item may be accessed directly at Extensions>annotatorApp.

NOTE:
  The menu item and the underlying Runnable do not depend on anything persistent in the script, except for a handle to the current PathObjectHierarchy reference. Although it has not been observed in testing, it is possible that if the root object of the hierarchy is changed, the hierarchy available to the tool would no longer correspond to the current GUI hierarchy. Possible workarounds include running the tool as a permanent (installed) extension with live access to the current QuPathGUI reference, implementing hooks to update the PathObjectSelectionModel reference, or simply re-running the script to generate a tool with reference to the active PathObjectHierarchy.

TODO:
  The menu item returned by the script is associated with a Runnable action, and can have its accelerator property specified to associate the menu item with a single keystroke. In this manner, the annotator can be run at a single keystroke upon annotating an object. The implementation of this varies by platform due to differences between representations of keystrokes, as well as the behavior of accelerators depending upon when they are added to the associated menu item. A keystroke implementation will be made available upon resolution of these discrepancies, and consultation with participants regarding their local platform, available QuPath version, and preferred keystrokes.