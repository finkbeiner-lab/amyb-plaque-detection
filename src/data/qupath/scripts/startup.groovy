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

def gui = QPEx.getQuPath().getInstance()
loadTileManager(gui)
loadContextMenu(gui)
