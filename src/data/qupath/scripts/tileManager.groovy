import groovy.transform.InheritConstructors

import java.io.BufferedReader
import java.io.FileReader
import java.io.File

import java.util.concurrent.Callable
import java.util.concurrent.FutureTask

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

import org.json.*
import org.apache.commons.io.FileUtils;

import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI

import qupath.lib.geom.Point2

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

import qupath.lib.roi.interfaces.ROI
import qupath.lib.roi.GeometryTools
import qupath.lib.roi.PolygonROI
import qupath.lib.roi.ROIs
import qupath.lib.roi.RoiTools

import qupath.lib.io.GsonTools


// TODO:
//  fix setSelected in hierarchyToTiles() - DONE
//  fix tile overlap calculation (check tiles are all overlapping the annotation) - DONE
//  clean up Params() semantics; import the new dialog/parameter classes from projects.groovy - DONE
//  determine better checks/handling for ROIs that are serializable
//  add ObservableList/ObservableMap
//  write tests for the (theoretically impossible) situation of having more than one labeled tile per location
//  add integration with the viewer grid overlay wrt. tile sizes?
//  stress-test the new implementation of the overlap calculation (will it fail for geometry-style ROIs?)
//  add helper to highlight/selectively move through the geometry ROIs to correct them before exporting/labelling in the workflow
//  distinguish between tiles partially vs tiles fully overlapping the locked tile grid when exporting


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

    def build() {
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


class FXUtils {
    static def alertCallable(String msg) {
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

    static def dialogCallable(DialogPane pane) {
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

    static def optionCallable(Map options) {
        def params = new Params()
        options.each({
            params.add(it.key, it.value.get(0), new OptionBox(it.value.get(1)))
        })
        def callable = FXUtils.dialogCallable(params.build())
        return new Callable<Map>() {
            @Override Map call() {
                if (!callable.call())
                    return null
                return params.paramDict.collectEntries({[it.key, it.value.get(1).getValue()]})
            }
        }

    }
}



class TileObjects implements Runnable {
    QuPathGUI gui
    PathObjectHierarchy hier

    Integer tileSize = 1024 // parameterize this?
    PathClass pathClass = PathClassFactory.getPathClass("tile") // allow recognizing subclasses of the base pathClass as well?
    ImagePlane imagePlane = ImagePlane.getPlane(0, 0) // do we still need this?
    Map<Integer, PathTileObject> tileMap = [:] // consider replacing or augmenting with an ObservableMap listening on the current hierarchy (must be sure that external routines do not make changes while we are running an op though)


    // Essential arithmetic/utility methods for computing object overlaps, tile coordinates & serialization, etc.

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

    List<Integer> getObjectTileOverlap(PathROIObject obj, boolean precise=false) {
        def roi = obj.getROI()
        def (xy, wh) = TileObjects.getROIBounds(roi)
        def tileCoords = this.getBoundsTileOverlap(*xy, *wh)
        if (precise && tileCoords.size() > 1) {
            def tileRegions = tileCoords.collect({ImageRegion.createInstance(it.get(0) * this.tileSize, it.get(1) * tileSize, this.tileSize, this.tileSize, roi.getZ(), roi.getT())})
            def tileOverlaps = tileRegions.collect({GeometryTools.regionToGeometry(it).intersects(roi.getGeometry())})
            tileCoords = tileCoords.withIndex().findAll({tileOverlaps.get(it.get(1))}).collect({it.get(0)})
        }
        return tileCoords.collect({TileObjects.encode(*it)})
    }


    def TileObjects(QuPathGUI gui) {
        this.gui = gui
    }


    // Returns true if there's a hierarchy to work with and parses valid tiles from the hierarchy

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


    // Parses valid tiles from the hierarchy into dictionary of tile id keys, tile object values

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
                  if (locked)
                      map.get(id).setLocked(locked)
                  if (selected)
                      selections.selectObjects([map.get(id)])
              }
          }
        })
        return map
    }


    // Creates a valid position-encoded tile object with specified lock status

    PathTileObject createTileObject(Integer id, Boolean locked=false) {
        def (x, y) = decode(id).collect({it * this.tileSize})
        def roi = ROIs.createRectangleROI(x, y, this.tileSize, this.tileSize, this.imagePlane)
        def obj = PathObjects.createTileObject(roi, this.pathClass, null)
        obj.setName(id.toString())
        obj.setLocked(locked)
        return obj
    }


    // Tile display managers; every change to tileMap goes through here and then updates the corresponding hierarchy components

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


    // Helper for computing tile/annotation overlaps in both directions
    //  "precise" refers to whether to compute overlaps via bounding box or whether to check that each tile also intersects the ROI

    List<Integer> tilesFromAnns(List<PathAnnotationObject> anns, boolean precise=false) {
        def tiles = []
        for (ann in anns) {
            for (tile in this.getObjectTileOverlap(ann, precise)) {
                if (!(tile in tiles)) {
                    tiles.add(tile)
                }
            }
        }
        return tiles
    }

    List<PathAnnotationObject> annsFromTiles(List<Integer> tiles, boolean precise=false) {
        def tiledAnns = []
        this.hier.getAnnotationObjects().each({
            for (tile in this.getObjectTileOverlap(it, precise)) {
                if (tile in tiles) {
                    if (!(it in tiledAnns)) {
                        tiledAnns.add(it)
                    }
                }
            }
        })
        return tiledAnns
    }


    // Helper for selectorDialog()

    void tileSelector(boolean select, boolean locked, boolean unlocked) {
        def sel = this.hier.getSelectionModel()
        def tiles = this.tileMap.values().findAll({(locked && it.isLocked()) || (unlocked && !it.isLocked())})
        if (select) {
            sel.selectObjects(tiles)
        } else {
            sel.deselectObjects(tiles)
        }
    }


    // Serialization implementation

    def jsonArrToList(JSONArray jsonArr) {
        return jsonArr.length() == 0 ? [] : (0 ..< jsonArr.length()).toArray().collect({jsonArr.get(it)})
    }

    JSONArray serializeAnns(List<PathAnnotationObject> anns, boolean precise=false) {
        // Warning: if the annotation does not have an annotation.getROI().getAllPoints() method, this will break
        return (new JSONArray(anns.collect({ def ann ->
            def roi = ann.getROI()
            def allPoints = roi.getAllPoints().collect({[it.getX(), it.getY()]})

            def map = [:]
            map.put("x", allPoints.collect({(double) it.get(0)}))
            map.put("y", allPoints.collect({(double) it.get(1)}))
            map.put("plane", [roi.getC(), roi.getZ(), roi.getT()])
            map.put("tiles", this.getObjectTileOverlap(ann, precise))
            map.put("pathClasses", ann.pathClass == null ? [] : PathClassTools.splitNames(ann.pathClass))
            return map
        })))
    }

    List deserializeAnns(String json, Integer classDepth=-1) {
        return jsonArrToList(new JSONArray(json)).collect({ JSONObject ann ->
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
        }).collect({ Map annDict ->
            def x = annDict.get("x")
            def y = annDict.get("y")
            def plane = ImagePlane.getPlaneWithChannel(*annDict.get("plane"))
            def roi = ROIs.createPolygonROI(x, y, plane)
            def pathClass = annDict.get("pathClasses").size() == 0 ? PathClassFactory.getPathClassUnclassified() : PathClassFactory.getPathClass(annDict.get("pathClasses"))
            def annObj = PathObjects.createAnnotationObject(roi, pathClass)
            return annObj
        })
    }


    // Serialization dialogs

    void exportJsonDialog() {
        def tiles = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
        def anns = this.annsFromTiles(tiles, true).findAll({it.getROI().getRoiType() == ROI.RoiType.AREA && it.getROI().getRoiName() == "Polygon"})

        def json = (new JSONObject())
            .put("annotations", this.serializeAnns(anns, true))
            .put("tiles", new JSONArray(tiles))

        def file = (new FileChooser()).showSaveDialog()
        if (file != null) {
            FileUtils.writeStringToFile(file, json.toString(), (String) null)
        }
    }

    void importJsonDialog() {
        def file = (new FileChooser()).showOpenDialog()
        if (file != null) {
            def json = new JSONObject(FileUtils.readFileToString(file, (String) null))
            def anns = deserializeAnns(json.get("annotations").toString())
            def tiles = jsonArrToList(json.get("tiles"))

            // if (this.alertCallable("Import " + anns.size().toString() " annotations for " + tiles.size().toString()).call()) {
            if (FXUtils.alertCallable("Import annotations?").call()) {
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

                def resp = FXUtils.dialogCallable(params.build()).call()
                def keepAnns = !resp ? false : params.get("anns").isSelected()
                def keepTiles = !resp ? false : params.get("tiles").isSelected()

                if (!keepAnns) {
                    this.hier.removeObjects(anns, false)
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
            if (FXUtils.dialogCallable(params.build()).call()) {
                def newJson = (new JSONObject())
                    .put("annotations", this.serializeAnns(anns, true))
                    .put("tiles", new JSONArray(tiles))
                def newFile = (new FileChooser()).showSaveDialog()
                if (newFile != null) {
                    FileUtils.writeStringToFile(newFile, newJson.toString(), null)
                }
            }
        }
    }


    // Main tile manipulation dialogs

    void builderDialog() {
        def sel = this.hier.getSelectionModel()
        def anns = this.hier.getAnnotationObjects()

        def params = new Params()
        params.add("type", "Filter annotations by: ", new OptionBox(["All", "Selected"]))

        params.add("lock", "Lock tiles: ", new CheckBox())
        params.get("lock").setSelected(false)
        params.get("lock").setAllowIndeterminate(false)

        if (FXUtils.dialogCallable(params.build()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.addTiles(this.tilesFromAnns(anns.findAll({resp == 0 || sel.isSelected(it)}), true), params.get("lock").isSelected())
            }
        }
    }

    void selectorDialog() {
        def params = new Params()
        params.add("type", "Filter tiles by: ", new OptionBox(["All", "Locked", "Unlocked"]))

        params.add("select", "Select tiles: ", new CheckBox())
        params.get("select").setSelected(true)
        params.get("select").setAllowIndeterminate(false)

        if (FXUtils.dialogCallable(params.build()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.tileSelector(params.get("select").isSelected(), resp == 0 || resp == 1, resp == 0 || resp == 2)
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

        if (FXUtils.dialogCallable(params.build()).call()) {
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

        if (FXUtils.dialogCallable(params.build()).call()) {
            def resp = params.get("type").getValue()
            if (resp != null) {
                this.removeTiles(this.tileMap.findAll({resp == 0 || sel.isSelected(it.value)}).collect({it.key}), params.get("lock").isSelected())
            }
        }
    }


    // Entrypoint

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

            if (FXUtils.dialogCallable(params.build()).call()) {
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
                        this.exportJsonDialog()
                    } else if (resp == 5) {
                        this.importJsonDialog()
                    } else if (resp == 6) {
                        this.modifyJsonDialog()
                    }
                }
            }
        } else {
            FXUtils.alertCallable("No ImageData available").call()
        }
    }

    static TileObjects getInstance(QuPathGUI gui) {
        return new TileObjects(gui)
    }
}


// def gui = QPEx.getQuPath().getInstance()
// gui.installCommand("Tile Manager", new TileObjects(gui))

return TileObjects
