import java.awt.image.BufferedImage
import javafx.beans.value.ChangeListener
import javafx.beans.value.ObservableValue
import qupath.lib.gui.scripting.DefaultScriptEditor
import qupath.lib.gui.QuPathGUI
import qupath.lib.projects.Project
import qupath.lib.projects.ResourceManager
import java.io.File

import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.gui.tools.GuiTools


def pathClassGroup = "plaque"
def pathClassMaps = [
    "plaque": [
        "Diffuse": [0, 255, 255],
        "Cored": [255, 153, 102],
        "CAA": [150, 79, 239],
        "Coarse-Grained":[160,170,90],
        "Cotton-Wool":[255,0,125],
        "Burned-Out":[124,125,255]
    ]
   ]
   
def prefMap = [
    "gridScaleMicrons": (boolean) true,
    "gridSpacingX": (double) 1024,
    "gridSpacingY": (double) 1024,
    "gridStartX": (double) 0,
    "gridStartY": (double) 0,

    "multipointTool": (boolean) false,
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


def setPathClasses(QuPathGUI gui, Map<String, List<Integer>> map) {
    GuiTools.runOnApplicationThread({
        gui.getAvailablePathClasses().setAll([
            PathClassFactory.getPathClassUnclassified(),
            *map.collect({
                assert it.value.size() == 6
                def pathClass = PathClassFactory.getPathClass(it.key)
                pathClass.setColor(ColorTools.packRGB(*it.value))
                return pathClass
            })
        ])
    })
}



println("Registered per project 'startup.groovy' handler")












