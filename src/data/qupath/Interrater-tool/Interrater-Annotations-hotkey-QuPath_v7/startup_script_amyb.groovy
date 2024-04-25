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
import javafx.application.Platform
import qupath.lib.gui.dialogs.Dialogs


def choice = Dialogs.showChoiceDialog("Select type", 
    "Please make your choice",
    ["Amyloid staining", "Biel","AT8"],
    "Amyloid staining")

print "You chose $choice"
def pathClassGroup = choice
def pathClassMaps = [
    "Amyloid staining": [
        "Cored:CTRL+C": [0, 255, 255],
        "Diffuse:CTRL+F": [255, 153, 102],
        //"CAA: CTRL+A": [150, 79, 239],
        "Coarse-Grained:CTRL+R":[255, 0, 239],
        "Cotton-Wool:CTRL+W":[160,170,90],
        "Burned-Out:CTRL+B":[124,125,255],
        "DELETE-Annotation:CTRL+D":[0,0,0]
    ],
   "AT8": [
        "Pre:CTRL+P": [0, 255, 0],
        "Mature:CTRL+M": [0, 0, 255],
        "Ghost:CTRL+G": [255, 0, 0],
        "DELETE-Annotation:CTRL+D":[0,0,0]
    ],
    "Biel":[
         "Cored:CTRL+C": [0, 255, 255],
        "Diffuse:CTRL+F": [255, 153, 102],
        //"CAA: CTRL+A": [150, 79, 239],
        "Coarse-Grained:CTRL+R":[255, 0, 239],
        "Cotton-Wool:CTRL+W":[160,170,90],
        "Burned-Out:CTRL+B":[124,125,255],
        "Pre:CTRL+P": [0, 255, 0],
        "Mature:CTRL+M": [0, 0, 255],
        "Ghost:CTRL+G": [255, 0, 0],
        "DELETE-Annotation:CTRL+D":[0,0,0]
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
    Platform.runLater({
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


//def setPathClasses1( Map<String, List<Integer>> map) {





gui = QuPathGUI.getInstance()
setPrefs(prefMap)
setPathClasses(gui, pathClassMaps.get(pathClassGroup))


     

gui.imageDataProperty().addListener(new ChangeListener<ImageData<BufferedImage>>() {
    @Override
    void changed(ObservableValue<? extends ImageData<BufferedImage>> observable, ImageData<BufferedImage> oldProject, ImageData<BufferedImage> newProject) {
        if (newProject == null || oldProject == newProject) {
            return
        }
        //def manager = newProject.getScripts()
        //print manager
        //def script
        //print newProject.currentLanguageProperty()
        try {
            def contextMenu = (new ScriptLoader()).getScript("interrater_annotator_hotkey_version.groovy")
        } catch (IOException ignored) {
            return
        }
        //DefaultScriptEditor.executeScript(DefaultScriptEditor.Language.GROOVY, contextMenu, newProject, null, true, null)
    }
})



println("Registered per project 'startup.groovy' handler")
