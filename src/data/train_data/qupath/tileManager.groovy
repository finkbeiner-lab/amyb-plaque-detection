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
import javafx.util.StringConverter

import java.util.concurrent.Callable
import java.util.concurrent.FutureTask

import org.json.*

import qupath.lib.geom.Point2




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

// class OptionBox {
//     OptionConverter converter
//     ChoiceBox box
//
//     class OptionConverter extends StringConverter<Integer> {
//         List<String> options
//         String defaultOption
//
//         def OptionConverter(List<String> options, String defaultOption=null) {
//             this.options = options
//             if (defaultOption == null) {
//                 this.defaultOption = ""
//             }
//         }
//
//         @Override String toString(Integer index) {
//             if (index == null) {
//                 return this.defaultOption
//             }
//             return this.options.get(index)
//         }
//
//         @Override Integer fromString(String value) {
//             return this.options.indexOf(value)
//         }
//     }
//
//     def OptionBox(List<String> options, String defaultOption=null) {
//         this.converter = new OptionConverter(options, defaultOption)
//         this.box = new ChoiceBox<Integer>()
//         this.box.setConverter(this.converter)
//         this.box.getItems().setAll((0 .. options.size() - 1).toArray())
//     }
// }

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



class TileObjects implements Runnable {
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


    def TileObjects(PathObjectHierarchy hier) {
        this.hier = hier
        this.hier.getTileObjects().each({
            if (it.getPathClass() == this.pathClass && it.getName() != null && it.getName().matches("[0-9]+")) {
                def id = it.getName().toInteger()
                if (!(id in this.tileMap.keySet()) || (it.isLocked() && !this.tileMap.get(id).isLocked())) {
                    this.tileMap.put(id, it)
                }
            }
        })
        this.renderTiles()
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

    void renderTiles() {
        this.hier.removeObjects(this.hier.getTileObjects(), false)
        this.hier.addPathObjects(this.tileMap.values())
    }

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



    // List<Integer> tilesFromSelectedAnns() {
    //     def sel = this.hier.getSelectionModel()
    //     def tiles = []
    //     this.hier.getAnnotationObjects()
    //         .findAll({sel.isSelected(it)})
    //         .each({
    //             for (tile in this.getObjectTileOverlap(it)) {
    //                 if (!(tile in tiles)) {
    //                     tiles.add(tile)
    //                 }
    //             }
    //         })
    //     return tiles
    // }

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



    void setSelectedTiles(boolean select, boolean locked, boolean unlocked) {
        def sel = this.hier.getSelectionModel()
        def tiles = this.tileMap.values().findAll({(locked && it.isLocked()) || (unlocked && !it.isLocked())})
        if (select) {
            sel.selectObjects(tiles)
        } else {
            sel.deselectObjects(tiles)
        }
    }


    // List<Integer> getSelectedTiles() {
    //   def sel = this.hier.getSelectionModel()
    //   return this.tileMap//.values()
    //   .findAll({sel.isSelected(it.value)})
    //   .collect({it.key})//getName().toInteger()})
    // }


    // void lockTiles(boolean lock) {
    //     def tiles = this.getSelectedTiles()
    //     if (this.alertCallable(tiles.toString()).call()) {
    //         tiles.collect({this.tileMap.get(it)}).each({it.setLocked(lock)})
    //     }
    // }

    // void removeUnlockedTiles() {
    //     def tiles = this.tileMap.findAll({!it.value.isLocked()}).collect({it.key})
    //     if (this.alertCallable(tiles.toString()).call()) {
    //         this.removeTiles(tiles)
    //     }
    // }


    def jsonArrToList(JSONArray jsonArr) {
      return jsonArr.length() == 0 ? [] : (0 .. jsonArr.length() - 1).toArray().collect({jsonArr.get(it)})
    }

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


    void getTilesAnnsJson() {
        def tiles = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
        def anns = this.annsFromTiles(tiles).findAll({it.getROI().getRoiType() == ROI.RoiType.AREA && it.getROI().getRoiName() == "Polygon"})

        String serialized = (new JSONObject())
            .put("annotations", this.serializeAnns(anns))
            .put("tiles", new JSONArray(tiles))
            .toString()

        def params = new Params()
        params.add("output", "JSON (output): ", new TextArea())
        params.get("output").setText(serialized)

        this.alertCallable("There are " + tiles.size().toString() + " locked tiles containing " + anns.size().toString() + " annotations.").call()
        this.dialogCallable(params.pane()).call()
    }

    void putTilesAnnsJson() {
        def params = new Params()
        params.add("input", "JSON (input): ", new TextArea())
        params.add("pc", "Use pathClass: ", new OptionBox(["None", "Base", "All"]))
        params.add("output", "JSON (output): ", new TextArea())

        if (this.dialogCallable(params.pane()).call()) {
            def pcRes = params.get("pc").getValue()
            def depth = -1
            if (pcRes != null && pcRes < 2) {
                depth = pcRes
            }

            def serialized = params.get("input").getText()
            def deserialized = new JSONObject(serialized)
            def anns = deserializeAnns(deserialized.get("annotations").toString(), depth)
            def tiles = (ArrayList<Integer>) jsonArrToList(deserialized.get("tiles"))

            params.get("output").setText(anns.toString())
            if (this.dialogCallable(params.pane()).call()) {
                this.hier.addPathObjects(anns)
                if (!this.alertCallable("Keep annotations?").call()) {
                    this.hier.removeObjects(anns, false)
                }
            }
        }
    }

    void modifyJson() {
        def params = new Params()
        params.add("input", "JSON (input): ", new TextArea())
        params.add("pc", "Use pathClass: ", new OptionBox(["None", "Base", "All"]))
        params.add("output", "JSON (output): ", new TextArea())

        if (this.dialogCallable(params.pane()).call()) {
            def pcRes = params.get("pc").getValue()
            def depth = -1
            if (pcRes != null && pcRes < 2) {
                depth = pcRes
            }

            def serialized = params.get("input").getText()
            def deserialized = new JSONObject(serialized)
            def anns = deserializeAnns(deserialized.get("annotations").toString(), depth)
            def tiles = (ArrayList<Integer>) jsonArrToList(deserialized.get("tiles"))


             String new_serialized = (new JSONObject())
            .put("annotations", this.serializeAnns(anns))
            .put("tiles", new JSONArray(tiles))
            .toString()

            params.get("output").setText(new_serialized)
             if (this.dialogCallable(params.pane()).call()) {
                this.hier.addPathObjects(anns)
                if (!this.alertCallable("Keep annotations?").call()) {
                    this.hier.removeObjects(anns, false)
                }
            }

        }
    }

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


    void run() {
        def optBox = new OptionBox([
            "Build tiles",
            "Select tiles",
            "Lock tiles",
            "Remove tiles",
            "Annotations -> Json",
            "Json -> Annotations",
            "modifyJson",
        ])

        def params = new Params()
        params.add("optBox", "Options: ", optBox)

        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("optBox").getValue()
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
                    this.getTilesAnnsJson()
                } else if (resp == 5) {
                    this.putTilesAnnsJson()
                } else if (resp == 6) {
                    this.modifyJson()
                }
            }
        }
    }
}




//def hier = QPEx.getCurrentHierarchy()
def gui = QPEx.getQuPath().getInstance()

//def app = new TileObjects(hier)
//gui.installCommand("tileTool", app)

class Runner implements Runnable {
    def Runner(QuPathGUI gui) {
        this.gui = gui
        this.imageData = null
        this.instance = null
    }

    def getInstance() {
        def imageData = this.gui.getImageData()
        if (imageData != null) {
            if (imageData != this.imageData) {
                this.imageData = imageData
                this.instance = new TileObjects(this.imageData.getHierarchy())
            } else if (this.instance == null) {
                this.instance = new TileObjects(this.imageData.getHierarchy())
            }
        } else {
            this.instance = null
        }
    }

    @Override void run() {
        this.getInstance()
        if (this.instance != null) {
            this.gui.installCommand("tileTool", this.instance)
        }
    }
}


gui.installCommand("tileToolBuilder", new Runnable() {
    @Override void run() {
        gui.installCommand("tileTool", new TileObjects(gui.getImageData().getHierarchy()))
    }
})
