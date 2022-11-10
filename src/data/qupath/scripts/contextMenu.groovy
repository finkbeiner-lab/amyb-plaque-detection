import java.util.prefs.Preferences

import javafx.beans.property.BooleanProperty
import javafx.beans.property.SimpleBooleanProperty

import javafx.scene.control.ContextMenu
import javafx.scene.control.Menu
import javafx.scene.control.MenuItem
import javafx.scene.input.MouseEvent
import javafx.event.EventHandler

import org.controlsfx.control.action.Action
import org.controlsfx.control.action.ActionUtils

import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.gui.viewer.QuPathViewer

import qupath.lib.gui.tools.GuiTools
import qupath.lib.gui.tools.MenuTools

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.hierarchy.events.PathObjectSelectionModel


class CustomPathPrefs {
    static String defaultRootNodeName = "/io.github.qupath/0.3"
    Preferences rootNode

    def CustomPathPrefs() {
        this(CustomPathPrefs.defaultRootNodeName)
    }

    def CustomPathPrefs(String rootNodeName) {
        this(rootNodeName, false)
    }

    def CustomPathPrefs(String rootNodeName, boolean create) {
        this.rootNode = CustomPathPrefs.getUserPrefsNode(rootNodeName, create)
        assert this.rootNode != null
    }

    static def getUserPrefsNode(String nodeName, boolean create) {
        def node = Preferences.userRoot()
        def nodeNames = nodeName.split("/")
        for (def name: nodeNames) {
            if (!create && !node.nodeExists(name))
                return null
            node = node.node(name)
        }
        return node
    }

    def getPrefs(String nodeName, String value) {
        return this.rootNode.get(nodeName, value)
    }

    def getPrefs(String nodeName, boolean value) {
        return this.rootNode.getBoolean(nodeName, value)
    }

    def setPrefs(String nodeName, String value) {
        this.rootNode.put(nodeName, value)
    }

    def setPrefs(String nodeName, boolean value) {
        this.rootNode.putBoolean(nodeName, value)
    }
}


class PathClassSetter {
    static boolean setClass(PathObject po, PathClass pc) {
        PathClass pcOld = po.getPathClass();
        po.setPathClass(pc);
        return po.getPathClass() != pcOld;
    }

    static List<PathObject> setClassChanged(List<PathObject> pos, PathClass pc) {
        List<PathObject> changed = new ArrayList<>();
        for (PathObject po: pos) {
            if (setClass(po, pc))
                changed.add(po);
        }
        return changed;
    }

    static void setSelectedAnnotationsClass(QuPathGUI instance, QuPathViewer viewer, PathClass pc) {
        if (viewer != null) {
            PathObjectHierarchy hier = viewer.getHierarchy();
            PathObjectSelectionModel sel = hier.getSelectionModel();
            List<PathObject> change = new ArrayList<>();
            for (PathObject po: hier.getAnnotationObjects()) {
                if (sel.isSelected(po))
                    change.add(po);
            }
            hier.fireObjectClassificationsChangedEvent(instance, setClassChanged(change, pc));
        }
    }
}


class PathClassIntensitySetter {
    static PathClass createIntensityClass(PathClass pc, int intensity) {
        pc = PathClassTools.getNonIntensityAncestorClass(pc);
        if (intensity == 0)
            return PathClassFactory.getNegative(pc);
        if (intensity == 1)
            return PathClassFactory.getOnePlus(pc);
        if (intensity == 2)
            return PathClassFactory.getTwoPlus(pc);
        if (intensity == 3)
            return PathClassFactory.getThreePlus(pc);
        if (intensity == 4)
            return PathClassFactory.getPositive(pc);
        return pc;
    }

    static boolean setIntensityClass(PathObject po, int intensity) {
        PathClass pc = po.getPathClass();
        po.setPathClass(createIntensityClass(pc, intensity));
        return po.getPathClass() != pc;
    }

    static List<PathObject> setIntensityClassChanged(List<PathObject> pos, int intensity) {
        List<PathObject> changed = new ArrayList<>();
        for (PathObject po: pos) {
            if (setIntensityClass(po, intensity))
                changed.add(po);
        }
        return changed;
    }

    static void setSelectedAnnotationsIntensityClass(QuPathGUI instance, QuPathViewer viewer, int intensity) {
        if (viewer != null) {
            PathObjectHierarchy hier = viewer.getHierarchy();
            PathObjectSelectionModel sel = hier.getSelectionModel();
            List<PathObject> change = new ArrayList<>();
            for (PathObject po: hier.getAnnotationObjects()) {
                if (sel.isSelected(po))
                    change.add(po);
            }
            hier.fireObjectClassificationsChangedEvent(instance, setIntensityClassChanged(change, intensity));
        }
    }
}


class PathClassIntensityContextMenu {
    static final String prefsNodeName = "annotationContextMenu"
    final CustomPathPrefs prefs

    final QuPathGUI gui
    final QuPathViewer viewer

    final ContextMenu contextMenu
    Menu menuSetClass
    Menu menuSetIntensity

    EventHandler<MouseEvent> handler


    def PathClassIntensityContextMenu(final QuPathGUI gui, final QuPathViewer viewer) {
        this.prefs = new CustomPathPrefs(CustomPathPrefs.defaultRootNodeName + "/" + PathClassIntensityContextMenu.prefsNodeName, true)

        this.gui = gui
        this.viewer = viewer

        this.contextMenu = new ContextMenu()
        this.menuSetClass = MenuTools.createMenu("Set class")
        this.menuSetIntensity = MenuTools.createMenu("Set intensity")
    }

    def setClassMenuItems() {
        if (this.viewer == null || !(this.viewer.getSelectedObject() instanceof PathAnnotationObject) || this.gui.getAvailablePathClasses().isEmpty()) {
            this.menuSetIntensity.getItems().clear()
            return
        }

        List<MenuItem> itemList = gui.getAvailablePathClasses().collect({it ->
            final PathClass pc = it.getName() == null ? null : it
            String pcName = pc == null ? "None" : pc.toString()
            Action action = new Action(pcName, e -> {PathClassSetter.setSelectedAnnotationsClass(this.gui, this.viewer, pc)})
            MenuItem item = ActionUtils.createMenuItem(action)
            return item
        })

        this.menuSetClass.getItems().setAll(itemList)
    }

    def setIntensityMenuItems() {
        if (this.viewer == null || !(this.viewer.getSelectedObject() instanceof PathAnnotationObject) || this.gui.getAvailablePathClasses().isEmpty()) {
            this.menuSetIntensity.getItems().clear()
            return
        }

        List<String> names = ["None", "Negative", "1+", "2+", "3+", "Positive"]
        List<MenuItem> itemList = names.withIndex().collect({name, idx ->
            final int i = idx - 1
            Action action = new Action(name, e -> {PathClassIntensitySetter.setSelectedAnnotationsIntensityClass(this.gui, this.viewer, i)})
            MenuItem item = ActionUtils.createMenuItem(action)
            return item
        })

        this.menuSetIntensity.getItems().setAll(itemList)
    }

    def setEventHandler() {
        EventHandler<MouseEvent> tmpHandler = new EventHandler<MouseEvent>() {
            @Override void handle(MouseEvent e) {
                if (PathClassIntensityContextMenu.this.prefs.getPrefs(PathClassIntensityContextMenu.this.viewer.hashCode().toString(), "") != this.hashCode().toString())
                    return

                if ((e.isPopupTrigger() || e.isSecondaryButtonDown()) && e.isShiftDown()) {
                    PathClassIntensityContextMenu.this.getContextMenu().show(PathClassIntensityContextMenu.this.viewer.getView().getScene().getWindow(), e.getScreenX(), e.getScreenY())
                    e.consume()
                }
            }
        }
        this.prefs.setPrefs(this.viewer.hashCode().toString(), tmpHandler.hashCode().toString())
        this.handler = tmpHandler
    }

    def getContextMenu() {
        this.setClassMenuItems()
        this.setIntensityMenuItems()

        List<MenuItem> menuItems = [this.menuSetClass, this.menuSetIntensity]
        this.contextMenu.getItems().setAll(menuItems)
        this.contextMenu.setAutoHide(true)

        return this.contextMenu
    }

    def build() {
        this.setEventHandler()
        this.viewer.getView().addEventFilter(MouseEvent.MOUSE_PRESSED, this.handler)
    }
}


class ContextMenuApp implements Runnable {
    QuPathGUI gui
    PathClassIntensityContextMenu contextMenu

    def ContextMenuApp(QuPathGUI gui) {
        this.gui = gui
    }

    void run() {
        this.gui.getViewers().each({(new PathClassIntensityContextMenu(gui, it).build())})
    }
}



def gui = QPEx.getQuPath().getInstance()
def menu = gui.installCommand("Extended Context Menu", new ContextMenuApp(gui))
