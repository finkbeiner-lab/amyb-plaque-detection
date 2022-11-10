import java.io.File

import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs


class ScriptLoader {
    static File scriptsPath = new File(new File(PathPrefs.userPathProperty().get()), "scripts")

    def ScriptLoader() {
        assert this.scriptsPath.isDirectory()
    }

    def getScriptText(String path) {
        def scriptPath = new File(this.scriptsPath, path)
        assert scriptPath.isFile()
        return scriptPath.getText()
    }

    def getScript(String path) {
        return Eval.me(this.getScriptText(path))
    }
}


def loadTileManager(QuPathGUI gui) {
    def tileManager = (new ScriptLoader()).getScript("tileManager.groovy").getInstance(gui)
    gui.installCommand("Tile Manager", tileManager)
}

def loadContextMenu(QuPathGUI gui) {
    def contextMenu = (new ScriptLoader()).getScript("contextMenu.groovy").getInstance(gui)
    gui.installCommand("Custom context menu", contextMenu)
    contextMenu.run()
}

def loadPathClasses(QuPathGUI gui, Map<String, List<Integer>> defaultPathClasses) {
    (new ScriptLoader()).getScript("setPathClasses.groovy").putPathClasses(gui, defaultPathClasses)
}


def gui = QPEx.getQuPath().getInstance()
def defaultPathClasses = [
    Neuritic: [0, 255, 0],
    Diffuse: [0, 255, 255],
    Core: [255, 153, 102],
    CAA: [150, 79, 239],
]

loadTileManager(gui)
loadContextMenu(gui)
loadPathClasses(gui, defaultPathClasses)
