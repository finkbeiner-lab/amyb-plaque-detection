import javafx.application.Platform

import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.classes.PathClassFactory


class PathClassHelper {
    static def fromRGB(int r, int g, int b) {
        return (r * 16 * 16) + (g * 16) + b
    }

    static def toRGB(int color) {
        int r = (color & 0xff0000) >> 16
        int g = (color & 0x00ff00) >> 8
        int b = (color & 0x0000ff)

        return [r, g, b]
    }

    static def getPathClasses(Map<String, List<Integer>> pathClasses) {
        return pathClasses.collect({
            assert it.value.size() == 3
            return PathClassFactory.getPathClass(it.key, fromRGB(*it.value))
        })
    }

    static def putPathClasses(QuPathGUI gui, Map<String, List<Integer>> pathClasses) {
        Platform.runLater({
            def pcList = gui.getAvailablePathClasses()
            pcList.setAll(PathClassFactory.getPathClassUnclassified())
            pcList.addAll(getPathClasses(pathClasses))
        })
    }
}


return PathClassHelper
