import static qupath.lib.gui.scripting.QPEx.*

hierarchy = getCurrentViewer().hierarchy
tileObjects = hierarchy.getTileObjects()
hierarchy.removeObjects(tileObjects, false)
