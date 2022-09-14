import groovy.transform.InheritConstructors

import qupath.lib.gui.scripting.QPEx

import qupath.lib.objects.PathObject
import qupath.lib.objects.PathRootObject
import qupath.lib.objects.PathROIObject
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathTileObject

import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathObjectTools

import qupath.lib.objects.classes.PathClass
import qupath.lib.objects.classes.PathClassTools
import qupath.lib.objects.classes.PathClassFactory

import qupath.lib.objects.hierarchy.PathObjectHierarchy

import qupath.lib.measurements.MeasurementList
import qupath.lib.measurements.MeasurementListFactory

import qupath.lib.regions.ImagePlane
import qupath.lib.regions.ImageRegion

import qupath.lib.roi.ROIs
import qupath.lib.roi.interfaces.ROI
import qupath.lib.roi.PolygonROI

import qupath.lib.io.GsonTools

import javafx.application.Platform
import javafx.concurrent.Task
import javafx.scene.control.Alert
import javafx.scene.control.Alert.AlertType
import javafx.scene.control.ButtonType
import javafx.scene.control.Dialog
import javafx.scene.control.DialogPane
import javafx.scene.control.CheckBox
import javafx.scene.control.ChoiceBox
import javafx.scene.control.Label
import javafx.scene.control.TextArea
import javafx.scene.control.TextField
import javafx.scene.layout.GridPane
import javafx.stage.Modality
import javafx.stage.FileChooser
import javafx.stage.DirectoryChooser
import javafx.util.StringConverter

import java.io.BufferedReader
import java.io.FileReader
import java.io.File


import java.util.concurrent.Callable
import java.util.concurrent.FutureTask

import org.json.*
import org.apache.commons.io.FileUtils;


import qupath.lib.geom.Point2



// Main wrapper class for storing/accessing parameters and generating GridPanes for Dialog boxes

class Params {
    def paramDict

    def Params() {
        this.paramDict = [:]
    }

    def add(String key, String label, Object param) {
        this.paramDict.put(key, [new Label(label), param])
    }

    def get(String key) {
        return this.paramDict.get(key)[1]
    }

    def pane() {
        def grid = new GridPane()
        grid.setHgap(10)
        grid.setVgap(10)
        this.paramDict.eachWithIndex({it, index -> grid.addRow(index, *it.value)})

        def pane = new DialogPane()
        pane.setContent(grid)
        pane.getButtonTypes().setAll(ButtonType.OK, ButtonType.CANCEL)
        return pane
    }
}


// Custom ChoiceBox wrapper to accept a fixed list of options and return option index number via converter by default

class OptionBox extends ChoiceBox<Integer> {
    class OptionConverter extends StringConverter<Integer> {
        List<String> options
        String defaultOption

        def OptionConverter(List<String> options, String defaultOption) {
            this.options = options
            this.defaultOption = defaultOption
        }

        @Override String toString(Integer index) {
            return index == null ? this.defaultOption : this.options.get(index)
        }

        @Override Integer fromString(String value) {
            return this.options.indexOf(value)
        }
    }

    def OptionBox(List<String> options) {
        this(options, new String())
    }

    def OptionBox(List<String> options, String defaultOption) {
        super()
        this.getItems().setAll((0 ..< options.size()).toArray())
        this.setConverter(new OptionConverter(options, defaultOption))
    }
}


// Main wrapper class for the app (need to break this up into controller/viewer classes)

class TileObjects implements Runnable {
    QuPathGUI gui
    PathObjectHierarchy hier

    Integer tileSize = 1024
    PathClass pathClass = PathClassFactory.getPathClass("tile")
    ImagePlane imagePlane = ImagePlane.getPlane(0, 0)
    Map<Integer, PathTileObject> tileMap = [:]

    static Integer round(Double x, Boolean floor=true) {
        if (floor) {
            return (x - (x - x.trunc() < 0 ? 1 : 0)).trunc().toInteger()
        }
        return -round(-x, true)
    }

    static Integer encode(Integer x, Integer y) {
        Integer z = x + y
        z *= z + 1
        z = z.intdiv(2)
        return y + z
    }

    static List<Integer> decode(Integer z) {
        Integer x = TileObjects.round(Math.sqrt((z * 8) + 1) - 1).intdiv(2)
        Integer y = z - encode(x, 0)
        return [x - y, y]
    }

    static List<List<Integer>> getROIBounds(ROI roi) {
        // Return the x, y, w, h bounds of an ROI
        Map map = [:]
        map.put({TileObjects.round(roi["bounds${it}"], true)}, ["X", "Y"])
        map.put({TileObjects.round(roi["bounds${it}"], false)}, ["Width", "Height"])
        return map.collect({k, v -> v.collect({k(it)})})
    }

    List<List<Integer>> getBoundsTileOverlap(Integer x, Integer y, Integer w, Integer h) {
        // Return the set of size t tiles which intersect an arbitrary rectangle ROI
        List<Integer> tiles = []
        for (Integer xi = x.intdiv(this.tileSize); xi <= (x + w).intdiv(this.tileSize); ++xi) {
            for (Integer yi = y.intdiv(this.tileSize); yi <= (y + h).intdiv(this.tileSize); ++yi) {
                tiles.add([xi, yi])
            }
        }
        return tiles
    }

    List<Integer> getObjectTileOverlap(PathROIObject obj) {
        def (xy, wh) = TileObjects.getROIBounds(obj.getROI())
        return this.getBoundsTileOverlap(*xy, *wh).collect({TileObjects.encode(*it)})
    }

    
    // Recent change: we now only persist a QuPathGUI instance, and on each call extract the active hierarchy if applicable and render tiles from there
    
    def TileObjects(QuPathGUI gui) {
        this.gui = gui
    }
    
    
    // TODO: make tileMap an ObservableMap?
    
    def loadTiles() {
        def imageData = this.gui.getImageData()
        if (imageData == null) {
            return false
        }

        this.hier = imageData.getHierarchy()
        this.tileMap = this.hierarchyToTiles(this.hier)
        this.renderTiles()
        return true
    }

    def hierarchyToTiles(PathObjectHierarchy hierarchy) {
        def map = [:]
        def selections = hierarchy.getSelectionModel()
        hierarchy.getTileObjects().each({
          if (it.getPathClass() == this.pathClass && it.getName() != null && it.getName().matches("[0-9]+")) {
              def id = it.getName().toInteger()
              def locked = it.isLocked()
              def selected = selections.isSelected(it)

              if (!(id in map.keySet())) {
                  map.put(id, it)
              } else {
                  map.get(id).setLocked(locked)
                  map.get(id).setSelected(selected)
              }
          }
        })
        return map
    }

    PathTileObject createTileObject(Integer id, Boolean locked=false) {
        def (x, y) = decode(id).collect({it * this.tileSize})
        def roi = ROIs.createRectangleROI(x, y, this.tileSize, this.tileSize, this.imagePlane)
        def obj = PathObjects.createTileObject(roi, this.pathClass, null)
        obj.setName(id.toString())
        obj.setLocked(locked)
        return obj
    }

    void addTiles(List<Integer> ids, Boolean locked=false) {
        ids.each({
            if (!(it in this.tileMap) || (locked && !this.tileMap.get(it).isLocked())) {
                this.tileMap.put(it, this.createTileObject(it, locked))
            }
        })
        this.renderTiles()
    }

    void removeTiles(List<Integer> ids, Boolean locked=false) {
        ids.each({
            if (it in this.tileMap.keySet() && (locked || !this.tileMap.get(it).isLocked())) {
                this.tileMap.remove(it)
            }
        })
        this.renderTiles()
    }
    
    
    // Refreshes the tile renders cheaply by holding on to tile objects and replacing with the authoritative copies of said objects from the tileMap
    // TODO: perform check on tile agreement here (i.e. loadTiles() call?) if we do not implement an ObservableMap for the tileMap
    
    void renderTiles() {
        this.hier.removeObjects(this.hier.getTileObjects(), false)
        this.hier.addPathObjects(this.tileMap.values())
    }

    
    // Utility for generating simple alert box
    def alertCallable(String msg) {
      return new Callable<Boolean>() {
          @Override Boolean call() {
              Alert alert = new Alert(AlertType.CONFIRMATION)
              alert.initModality(Modality.NONE)
              alert.setContentText(msg)
              Optional<ButtonType> resp = alert.showAndWait()
              return resp.isPresent() && resp.get() == ButtonType.OK
          }
      }
    }
    
    // Utility for generating Dialog box with custom GridPane inserted and modality None (should this be a parameter?)
    def dialogCallable(DialogPane pane) {
        return new Callable<Boolean>() {
            @Override Boolean call() {
                Dialog dialog = new Dialog()
                dialog.initModality(Modality.NONE)
                dialog.setDialogPane(pane)
                Optional<ButtonType> resp = dialog.showAndWait()
                return resp.isPresent() && resp.get() == ButtonType.OK
            }
        }
    }


    List<Integer> tilesFromAnns(List<PathAnnotationObject> anns) {
        def tiles = []
        for (ann in anns) {
            for (tile in this.getObjectTileOverlap(ann)) {
                if (!(tile in tiles)) {
                    tiles.add(tile)
                }
            }
        }
        return tiles
    }

    List<PathAnnotationObject> annsFromTiles(List<Integer> tiles) {
        def tiledAnns = []
        this.hier.getAnnotationObjects().each({
            for (tile in this.getObjectTileOverlap(it)) {
                if (tile in tiles) {
                    if (!(it in tiledAnns)) {
                        tiledAnns.add(it)
                    }
                }
            }
        })
        return tiledAnns
    }


    // Select: whether to select or deselect, Locked: whether to address locked tiles, Unlocked: whether to address unlocked tiles
    void setSelectedTiles(boolean select, boolean locked, boolean unlocked) {
        def sel = this.hier.getSelectionModel()
        def tiles = this.tileMap.values().findAll({(locked && it.isLocked()) || (unlocked && !it.isLocked())})
        if (select) {
            sel.selectObjects(tiles)
        } else {
            sel.deselectObjects(tiles)
        }
    }


    def jsonArrToList(JSONArray jsonArr) {
      return jsonArr.length() == 0 ? [] : (0 ..< jsonArr.length()).toArray().collect({jsonArr.get(it)})
    }

    
    // Serialization routines:
    //   should these be broken out into a separate class for tweaks and for testing/portability purposes?
    
    
    JSONArray serializeAnns(List<PathAnnotationObject> anns) {
        return (new JSONArray(anns.collect({ def ann ->
            def roi = ann.getROI()
            def allPoints = roi.getAllPoints().collect({[it.getX(), it.getY()]})

            def map = [:]
            map.put("x", allPoints.collect({(double) it.get(0)}))
            map.put("y", allPoints.collect({(double) it.get(1)}))
            map.put("plane", [roi.getC(), roi.getZ(), roi.getT()])
            map.put("tiles", this.getObjectTileOverlap(ann))
            map.put("pathClasses", ann.pathClass == null ? [] : PathClassTools.splitNames(ann.pathClass))
            return map
        })))
    }

    List deserializeAnns(String json, Integer classDepth=-1) {
        def annsArr = jsonArrToList(new JSONArray(json))

        annsArr = annsArr.collect({ JSONObject ann ->
            def map = [:]

            map.put("x", (double[]) jsonArrToList(ann.get("x")))
            map.put("y", (double[]) jsonArrToList(ann.get("y")))
            map.put("tiles", (ArrayList<Integer>) jsonArrToList(ann.get("tiles")))//.collect({(Integer) it}))
            map.put("plane", (ArrayList<Integer>) jsonArrToList(ann.get("plane")))//.collect({(Integer) it}))
            map.put("pathClasses", (ArrayList<String>) jsonArrToList(ann.get("pathClasses")))//.collect({(String) it}))

            if (classDepth != -1 && map.get("pathClasses").size() > classDepth) {
                map.put("pathClasses", map.get("pathClasses").subList(0, classDepth))
            }

            return map
        })

        return annsArr.collect({ Map annDict ->
            def x = annDict.get("x")
            def y = annDict.get("y")
            def plane = ImagePlane.getPlaneWithChannel(*annDict.get("plane"))
            def roi = ROIs.createPolygonROI(x, y, plane)
            def pathClass = annDict.get("pathClasses").size() == 0 ? PathClassFactory.getPathClassUnclassified() : PathClassFactory.getPathClass(annDict.get("pathClasses"))
            def annObj = PathObjects.createAnnotationObject(roi, pathClass)
            return annObj
        })
    }

    // Old text-based serialization dialogs
    
    // void getTilesAnnsJson() {
    //     def tiles = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
    //     def anns = this.annsFromTiles(tiles).findAll({it.getROI().getRoiType() == ROI.RoiType.AREA && it.getROI().getRoiName() == "Polygon"})
    //
    //     String serialized = (new JSONObject())
    //         .put("annotations", this.serializeAnns(anns))
    //         .put("tiles", new JSONArray(tiles))
    //         .toString()
    //
    //     def params = new Params()
    //     params.add("output", "JSON (output): ", new TextArea())
    //     params.get("output").setText(serialized)
    //
    //     this.alertCallable("There are " + tiles.size().toString() + " locked tiles containing " + anns.size().toString() + " annotations.").call()
    //     this.dialogCallable(params.pane()).call()
    // }
    //
    // void putTilesAnnsJson() {
    //     def params = new Params()
    //     params.add("input", "JSON (input): ", new TextArea())
    //     params.add("pc", "Use pathClass: ", new OptionBox(["None", "Base", "All"]))
    //     params.add("output", "JSON (output): ", new TextArea())
    //
    //     if (this.dialogCallable(params.pane()).call()) {
    //         def pcRes = params.get("pc").getValue()
    //         def depth = -1
    //         if (pcRes != null && pcRes < 2) {
    //             depth = pcRes
    //         }
    //
    //         def serialized = params.get("input").getText()
    //         def deserialized = new JSONObject(serialized)
    //         def anns = deserializeAnns(deserialized.get("annotations").toString(), depth)
    //         def tiles = (ArrayList<Integer>) jsonArrToList(deserialized.get("tiles"))
    //
    //         params.get("output").setText(anns.toString())
    //         if (this.dialogCallable(params.pane()).call()) {
    //             this.hier.addPathObjects(anns)
    //             if (!this.alertCallable("Keep annotations?").call()) {
    //                 this.hier.removeObjects(anns, false)
    //             }
    //         }
    //     }
    // }
    //
    // void modifyJson() {
    //     def params = new Params()
    //     params.add("input", "JSON (input): ", new TextArea())
    //     params.add("pc", "Use pathClass: ", new OptionBox(["None", "Base", "All"]))
    //     params.add("output", "JSON (output): ", new TextArea())
    //
    //     if (this.dialogCallable(params.pane()).call()) {
    //         def pcRes = params.get("pc").getValue()
    //         def depth = -1
    //         if (pcRes != null && pcRes < 2) {
    //             depth = pcRes
    //         }
    //
    //         def serialized = params.get("input").getText()
    //         def deserialized = new JSONObject(serialized)
    //         def anns = deserializeAnns(deserialized.get("annotations").toString(), depth)
    //         def tiles = (ArrayList<Integer>) jsonArrToList(deserialized.get("tiles"))
    //
    //
    //          String new_serialized = (new JSONObject())
    //         .put("annotations", this.serializeAnns(anns))
    //         .put("tiles", new JSONArray(tiles))
    //         .toString()
    //
    //         params.get("output").setText(new_serialized)
    //          if (this.dialogCallable(params.pane()).call()) {
    //             this.hier.addPathObjects(anns)
    //             if (!this.alertCallable("Keep annotations?").call()) {
    //                 this.hier.removeObjects(anns, false)
    //             }
    //         }
    //
    //     }
    // }

    
    // New file-based serialization dialogs
    
    void exportJsonDialog() {
        def tiles = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
        def anns = this.annsFromTiles(tiles).findAll({it.getROI().getRoiType() == ROI.RoiType.AREA && it.getROI().getRoiName() == "Polygon"})

        def json = (new JSONObject())
            .put("annotations", this.serializeAnns(anns))
            .put("tiles", new JSONArray(tiles))

        def file = (new FileChooser()).showSaveDialog()
        if (file != null) {
            FileUtils.writeStringToFile(file, json.toString(), null)
        }
    }

    void importJsonDialog() {
        def file = (new FileChooser()).showOpenDialog()
        if (file != null) {
            def json = new JSONObject(FileUtils.readFileToString(file, null))
            def anns = deserializeAnns(json.get("annotations").toString())
            def tiles = jsonArrToList(json.get("tiles"))

            // if (this.alertCallable("Import " + anns.size().toString() " annotations for " + tiles.size().toString()).call()) {
            if (this.alertCallable("Import annotations?").call()) {
                def oldTilesLocked = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
                def oldTilesUnlocked = this.tileMap.findAll({!it.value.isLocked()}).collect({it.key})

                this.hier.addPathObjects(anns)
                this.addTiles(tiles, true)

                def params = new Params()
                params.add("anns", "Keep annotations: ", new CheckBox())
                params.get("anns").setSelected(true)
                params.get("anns").setAllowIndeterminate(false)

                params.add("tiles", "Keep locked tiles: ", new CheckBox())
                params.get("tiles").setSelected(true)
                params.get("tiles").setAllowIndeterminate(false)

                def resp = this.dialogCallable(params.pane()).call()
                def keepAnns = !resp ? false : params.get("anns").isSelected()
                def keepTiles = !resp ? false : params.get("tiles").isSelected()

                if (!keepAnns) {
                    this.hier.removeObjects(anns)
                }
                if (!keepTiles) {
                    this.removeTiles(tiles, true)
                    this.addTiles(oldTilesLocked, true)
                    this.addTiles(oldTilesUnlocked, false)
                }
            }
        }
    }

    void modifyJsonDialog() {
        def oldFile = (new FileChooser()).showOpenDialog()
        if (oldFile != null) {
            def oldJson = new JSONObject(FileUtils.readFileToString(oldFile, null))
            def anns = deserializeAnns(oldJson.get("annotations").toString())
            def tiles = jsonArrToList(oldJson.get("tiles"))

            def params = new Params()
            params.add("pc", "Use pathClass: ", new OptionBox(["None", "Base", "All"]))
            if (this.dialogCallable(params.pane()).call()) {
                def newJson = (new JSONObject())
                    .put("annotations", this.serializeAnns(anns))
                    .put("tiles", new JSONArray(tiles))
                def newFile = (new FileChooser()).showSaveDialog()
                if (newFile != null) {
                    FileUtils.writeStringToFile(newFile, newJson.toString(), null)
                }
            }
        }
    }

    
    // New builder, selector, locker, remover Dialogs

    void builderDialog() {
        def sel = this.hier.getSelectionModel()
        def anns = this.hier.getAnnotationObjects()

        def params = new Params()
        params.add("type", "Filter annotations by: ", new OptionBox(["All", "Selected"]))

        params.add("lock", "Lock tiles: ", new CheckBox())
        params.get("lock").setSelected(false)
        params.get("lock").setAllowIndeterminate(false)

        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.addTiles(this.tilesFromAnns(anns.findAll({resp == 0 || sel.isSelected(it)})), params.get("lock").isSelected())
            }
        }
    }

    void selectorDialog() {
        def params = new Params()
        params.add("type", "Filter tiles by: ", new OptionBox(["All", "Locked", "Unlocked"]))

        params.add("select", "Select tiles: ", new CheckBox())
        params.get("select").setSelected(true)
        params.get("select").setAllowIndeterminate(false)

        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.setSelectedTiles(params.get("select").isSelected(), resp == 0 || resp == 1, resp == 0 || resp == 2)
            }
        }
    }

    void lockerDialog() {
        def sel = this.hier.getSelectionModel()

        def params = new Params()
        params.add("type", "Filter tiles by: ", new OptionBox(["All", "Selected"]))

        params.add("lock", "Lock tiles: ", new CheckBox())
        params.get("lock").setSelected(true)
        params.get("lock").setAllowIndeterminate(false)

        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                def lock = params.get("lock").isSelected()
                this.tileMap.findAll({resp == 0 || sel.isSelected(it.value)}).each({it.value.setLocked(lock)})
            }
        }
    }

    void removerDialog() {
        def sel = this.hier.getSelectionModel()

        def params = new Params()
        params.add("type", "Filter tiles by: ", new OptionBox(["All", "Selected"]))

        params.add("lock", "Remove locked tiles: ", new CheckBox())
        params.get("lock").setSelected(false)
        params.get("lock").setAllowIndeterminate(false)

        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.removeTiles(this.tileMap.findAll({resp == 0 || sel.isSelected(it.value)}).collect({it.key}), params.get("lock").isSelected())
            }
        }
    }

    
    // This is called when you click the menu
    
    void run() {
        if (this.loadTiles()) {
            def params = new Params()
            params.add("options", "Options: ", new OptionBox([
                "Build tiles",
                "Select tiles",
                "Lock tiles",
                "Remove tiles",
                "Export to JSON",
                "Import from JSON",
                "Modify JSON",
            ]))

            if (this.dialogCallable(params.pane()).call()) {
                def resp = params.get("options").getValue()
                if (resp != null) {
                    if (resp == 0) {
                        this.builderDialog()
                    } else if (resp == 1) {
                        this.selectorDialog()
                    } else if (resp == 2) {
                        this.lockerDialog()
                    } else if (resp == 3) {
                        this.removerDialog()
                    } else if (resp == 4) {
                        // this.getTilesAnnsJson()
                        this.exportJsonDialog()
                    } else if (resp == 5) {
                        // this.putTilesAnnsJson()
                        this.importJsonDialog()
                    } else if (resp == 6) {
                        // this.modifyJson()
                        this.modifyJsonDialog()
                    }
                }
            }
        } else {
            this.alertCallable("No ImageData available").call()
        }
    }
}


def gui = QPEx.getQuPath().getInstance()
gui.installCommand("Tile Manager", new TileObjects(gui))



// Old workaround hack to prevent old cache from carrying over to a new image context (obsolete)

// class Runner implements Runnable {
//     QuPathGUI gui
//     ImageData imageData
//     Runnable instance
//
//     def Runner(QuPathGUI gui) {
//         this.gui = gui
//         this.imageData = null
//         this.instance = null
//     }
//
//     def getInstance() {
//         def imageData = this.gui.getImageData()
//         if (imageData == null) {
//             this.instance = null
//         } else if (imageData != this.imageData) {
//             this.instance = new TileObjects(imageData.getHierarchy())
//         }
//         this.imageData = imageData
//         return this.instance
//     }
//
//     @Override void run() {
//         if (this.getInstance() != null) {
//             this.instance.run()
//         }
//     }
// }
//
//
// def gui = QPEx.getQuPath().getInstance()
// gui.installCommand("Tile Manager", new Runner(gui))
