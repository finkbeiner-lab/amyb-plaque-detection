import groovy.transform.InheritConstructors

import qupath.lib.gui.scripting.QPEx

import qupath.lib.objects.PathObject // PathObject base class, subclasses
import qupath.lib.objects.PathRootObject
import qupath.lib.objects.PathROIObject
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathTileObject

import qupath.lib.objects.PathObjects // PathObject constructor entrypoints
import qupath.lib.objects.PathObjectTools // PathObject predicate creation, management, utilities

import qupath.lib.objects.classes.PathClass // PathClass base class
import qupath.lib.objects.classes.PathClassFactory // PathClass constructor entrypoints, StandardPathClasses, PathClass utilities

import qupath.lib.objects.hierarchy.PathObjectHierarchy // PathObjectHierarchy base class

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
import javafx.scene.control.ChoiceBox
import javafx.scene.control.Label
import javafx.scene.control.TextArea
import javafx.scene.control.TextField
import javafx.scene.layout.GridPane
import javafx.stage.Modality
import javafx.util.StringConverter

import java.util.concurrent.Callable
import java.util.concurrent.FutureTask



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

class OptionConverter extends StringConverter<Integer> {
    List<String> opts
    def OptionConverter(List<String> opts) {
        this.opts = opts
    }
    String toString(Integer value) {
        return this.opts.get(value)
    }
    Integer fromString(String value) {
        return this.opts.indexOf(value)
    }
}




class TileObjects implements Runnable {
    PathObjectHierarchy hier
    
    Integer tileSize = 1024
    PathClass pathClass = PathClassFactory.getPathClass("tile")
    ImagePlane imagePlane = ImagePlane.getPlane(0, 0)
    Map<Integer, PathTileObject> tileMap = [:]
    
    Integer runId
    
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
        this.runId = runId
        this.hier.getTileObjects().each({
            if (it.getPathClass() == this.pathClass && it.getName() != null && it.getName().matches("[0-9]+")) {
                def id = it.getName().toInteger()
                if (!(id in this.tileMap.keySet() && this.tileMap.get(id).isLocked() && !it.isLocked())) {
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
        return new Callable<Optional<ButtonType>>() {
            @Override Optional<ButtonType> call() {
                Alert alert = new Alert(AlertType.CONFIRMATION)
                alert.initModality(Modality.NONE)
                alert.setContentText(msg)
                return alert.showAndWait()
            }
        }
    }
    
    def dialogCallable(DialogPane pane) {
        return new Callable<Boolean>() {
            @Override Boolean call() {
                Dialog dialog = new Dialog()
                dialog.setDialogPane(pane)
                Optional<ButtonType> resp = dialog.showAndWait()
                return resp.isPresent() && resp.get() == ButtonType.OK
            }
        }
    }
    
    List<Integer> tileSelector() {
        def sel = this.hier.getSelectionModel()
        return this.tileMap.values()
            .findAll({sel.isSelected(it)})
            .collect({it.getName().toInteger()})
    }
    
    List<Integer> annSelector() {
        def sel = this.hier.getSelectionModel()
        def tiles = []
        this.hier.getAnnotationObjects()
            .findAll({sel.isSelected(it)})
            .each({
                for (tile in this.getObjectTileOverlap(it)) {
                    if (!(tile in tiles)) {
                        tiles.add(tile)
                    }
                }
            })
        return tiles
    }
    
    List<PathAnnotationObject> tiledAnnSelector(List<Integer> tiles) {
        def tiledAnns = []
//        def tilesToAnns = [:]
        this.hier.getAnnotationObjects()
            .each({
                for (tile in this.getObjectTileOverlap(it)) {
                    if (tile in tiles) {
                        if (!(it in tiledAnns)) {
                            tiledAnns.add(it)
                        }
//                        if (!(tile in tilesToAnns.keySet())) {
//                            tilesToAnns.put(tile, [])
//                        }
//                        tilesToAnns.get(tile).add(it)
                    }
                }
            })
        return tiledAnns//, tilesToAnns
    }
    
    
    void buildUnlockedTiles() {
        def tiles = this.annSelector()
        def res = this.alertCallable(tiles.collect({this.decode(it)}).toString()).call()
        if (res.isPresent() && res.get() == ButtonType.OK) {
            this.addTiles(tiles, false)
        }
    }
    
    void lockTiles(boolean lock) {
        def tiles = this.tileSelector()
        def res = this.alertCallable(tiles.toString()).call()
        if (res.isPresent() && res.get() == ButtonType.OK) {
            tiles.collect({this.tileMap.get(it)}).each({it.setLocked(lock)})
        }
    }
    
    void removeUnlockedTiles() {
        def tiles = this.tileMap.findAll({!it.value.isLocked()}).collect({it.key})
        def res = this.alertCallable(tiles.toString()).call()
        if (res.isPresent() && res.get() == ButtonType.OK) {
            this.removeTiles(tiles)
        }
    }
    
    void getLockedTileAnns() {
        def tiles = this.tileMap.findAll({it.value.isLocked()}).collect({it.key})
        def tiledAnns = this.tiledAnnSelector(tiles).findAll({it.getROI().getRoiType() == ROI.RoiType.AREA && it.getROI().getRoiName() == "Polygon"})
        def annDict = tiledAnns.collect({
            [
                Points: it.getROI().getAllPoints().collect({point -> [point.getX(), point.getY()].collect({it.toInteger()})}),
                Tiles: this.getObjectTileOverlap(it).collect({this.decode(it)}),
            ]
        })
        this.alertCallable("There are " + tiledAnns.size().toString() + " annotations for " + tiles.size().toString() + " locked tiles.").call()
        
        def params = new Params()
        params.add("display", "JSON: ", new TextArea())
        params.get("display").setText(annDict.toString())
        this.dialogCallable(params.pane()).call()
    }
    

    
    void run() {
        this.runId = 0
        

        
        def optList = [
            "Build tiles (from annotations)",
            "Lock tiles",
            "Unlock tiles",
            "Remove unlocked tiles",
            "Serialize annotations (from locked tiles)"
        ]
        def opts = new ChoiceBox()
        opts.getItems().setAll(optList)
        
        def params = new Params()
        params.add("opts", "Options: ", opts)
        
        if (this.dialogCallable(params.pane()).call()) {
            def resp = params.get("opts").getValue()
            if (resp != null) {
                def opt = optList.indexOf(resp)
                this.alertCallable(opt.toString()).call()
                
                if (opt == 0) {
                    this.buildUnlockedTiles()
                } else if (opt == 1) {
                    this.lockTiles(true)
                } else if (opt == 2) {
                    this.lockTiles(false)
                } else if (opt == 3) {
                    this.removeUnlockedTiles()
                } else if (opt == 4) {
                    this.getLockedTileAnns()
                }
            }
        }
    }
}


def hier = QPEx.getCurrentHierarchy()
def gui = QPEx.getQuPath().getInstance()

def app = new TileObjects(hier)
gui.installCommand("tileTool", app)
