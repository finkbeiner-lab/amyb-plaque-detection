import java.io.File

import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.gui.tools.GuiTools
import qupath.lib.gui.viewer.OverlayOptions


def pathClassGroup = "plaque"
def pathClassMaps = [
    "plaque": [
        "Neuritic": [0, 255, 0],
        "Diffuse": [0, 255, 255],
        "Core": [255, 153, 102],
        "CAA": [150, 79, 239],
    ],

    "tangle": [
        "Pre": [0, 255, 0],
        "Mature": [0, 255, 255],
        "Ghost": [255, 153, 102],
    ],
]

def prefMap = [
    "gridScaleMicrons": (boolean) false,
    "gridSpacingX": (double) 1024,
    "gridSpacingY": (double) 1024,
    "gridStartX": (double) 0,
    "gridStartY": (double) 0,

    "multipointTool": (boolean) false,
]

def overlayOptionsMap = [
    "fillDetections": (boolean) true,
    "opacity": (float) 1,
    "showAnnotations": (boolean) true,
    "showDetections": (boolean) true,
    "showGrid": (boolean) true,
    "showPixelClassification": (boolean) true,
    "showTMAGrid": (boolean) true,
]


class ScriptLoader {
    static File scriptsPath = new File(new File(PathPrefs.userPathProperty().get()), "scripts")

    def ScriptLoader() {
        assert this.scriptsPath.isDirectory()
    }

    def getScriptFile(String path) {
        return new File(this.scriptsPath, path)
    }

    def getScript(String path) {
        def scriptFile = this.getScriptFile(path)
        assert scriptFile.isFile()
        return Eval.me(scriptFile.getText())
    }

    def getScriptOptional(String path) {
        def scriptFile = this.getScriptFile(path)
        return scriptFile.isFile() ? Eval.me(scriptFile.getText()) : null
    }
}


def setPrefs(Map<String, Object> map) {
    return map
        .collectEntries({[it.key, PathPrefs.@"${it.key}"]})
        .findAll({it.value.get() != map.get(it.key)})
        .each({it.value.set(map.get(it.key))})
}

def setOverlayOptions(OverlayOptions overlayOptions, Map<String, Object> map) {
    return GuiTools.callOnApplicationThread({
        return map
            .collectEntries({[it.key, overlayOptions.@"${it.key}"]})
            .findAll({it.value.get() != map.get(it.key)})
            .each({it.value.set(map.get(it.key))})
    })
}

def setPathClasses(QuPathGUI gui, Map<String, List<Integer>> map) {
    GuiTools.runOnApplicationThread({
        gui.getAvailablePathClasses().setAll([
            PathClassFactory.getPathClassUnclassified(),
            *map.collect({
                assert it.value.size() == 3
                def pathClass = PathClassFactory.getPathClass(it.key)
                pathClass.setColor(ColorTools.packRGB(*it.value))
                return pathClass
            })
        ])
    })
}


def loadContextMenu(QuPathGUI gui) {
    def contextMenu = (new ScriptLoader()).getScript("contextMenu.groovy").getInstance(gui)
    gui.installCommand("Custom context menu", contextMenu)
    contextMenu.run()
    return true
}

def loadTileManager(QuPathGUI gui) {
    def tileManager = (new ScriptLoader()).getScript("tileManager.groovy").getInstance(gui)
    gui.installCommand("Tile Manager", tileManager)
    return true
}


def gui = QPEx.getQuPath().getInstance()

setPrefs(prefMap)
setOverlayOptions(gui.getOverlayOptions(), overlayOptionsMap)
setPathClasses(gui, pathClassMaps.get(pathClassGroup))

loadContextMenu(gui)
loadTileManager(gui)
