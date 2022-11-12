import java.io.File
import qupath.lib.gui.prefs.PathPrefs
Eval.me(new File(new File(PathPrefs.userPathProperty().get()), "scripts/startup.groovy").getText())
