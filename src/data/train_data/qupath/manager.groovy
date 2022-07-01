import qupath.lib.gui.scripting.QPEx
import static qupath.lib.gui.scripting.QPEx.*

import qupath.lib.objects.PathObject // PathObject base class, subclasses
import qupath.lib.objects.PathObjects // PathObject constructor entrypoints
import qupath.lib.objects.PathObjectTools // PathObject predicate creation, management, utilities
import qupath.lib.objects.classes.PathClass // PathClass base class
import qupath.lib.objects.classes.PathClassFactory // PathClass constructor entrypoints, StandardPathClasses, PathClass utilities
import qupath.lib.objects.hierarchy.PathObjectHierarchy // PathObjectHierarchy base class

import qupath.lib.objects.PathRootObject
import qupath.lib.objects.PathROIObject
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathDetectionObject
import qupath.lib.objects.PathTileObject

import qupath.lib.measurements.MeasurementList
import qupath.lib.measurements.MeasurementListFactory

import qupath.lib.roi.ROIs
import qupath.lib.roi.interfaces.ROI
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.ImageRegion
import qupath.lib.io.GsonTools



Integer round(Double x, Boolean floor=true) {
    if (floor) {
        return (x - (x - x.trunc() < 0 ? 1 : 0)).trunc().toInteger()
    }
    return -round(-x, true)
}

def mapOverLists = {Map<Object, List<Object>> a -> a.collect({k, v -> v.collect({k(it)})})}


def pathObjectMap(List<PathObject> objects) {
    def unlabeled = []
    def labeled = [:]
    for (object in objects) {
        pathClass = object.pathClass
        if (pathClass == null) {
            unlabeled.add(object)
        } else {
            if (!labeled.containsKey(pathClass)) {
                labeled.put(pathClass, [])
            }
            labeled.get(pathClass).add(object)
        }
    }
    return [unlabeled, labeled]
}

def removePathObject(PathObject object) {
    object.getParent().removePathObject(object)
}


def pathObjectsToROIs = {List<PathROIObject> objs -> objs.collect({it.getROI()})}


def getROIBounds(ROI roi) {
    def map = [:]
    map.put({round(roi["bounds${it}"], floor=true)}, ["X", "Y"])
    map.put({round(roi["bounds${it}"], floor=false)}, ["Width", "Height"])
    return map.collect({k, v -> v.collect({k(it)})})
}

def getTileBoundsOverlap(Integer t, Integer x, Integer y, Integer w, Integer h) {
    List<List<Integer>> tiles = []
    for (xi = x.intdiv(t); xi <= (x + w).intdiv(t); ++xi) {
        for (yi = y.intdiv(t); yi <= (y + h).intdiv(t); ++yi) {
            tiles.add([xi, yi])
        }
    }
    return tiles
}


def pathObjectsToTiles = {Integer t, List<PathROIObject> objects ->
    objects.collect({
        it.getROI()
    }).collect({
        getROIBounds(it)
    }).collect({
        [it[0][0], it[0][1], it[1][0], it[1][1]]
    }).collect({
        getTileBoundsOverlap(t, *it)
    })
}

def pathObjectsTileMap = {Integer t, List<PathROIObject> objects ->
    def tiles = [:]
    for (object in objects) {
        def objBounds = getROIBounds(object.getROI())
        def objTiles = getTileBoundsOverlap(t, *objBounds[0], *objBounds[1])
        for (tile in objTiles) {
            if (!tiles.containsKey(tile)) {
                tiles.put(tile, [])
            }
            tiles.get(tile).add(object)
        }
    }
    return tiles
}

def createTileROI = {Integer t, Integer x, Integer y, ImagePlane plane ->
    ROIs.createRectangleROI(x * t, y * t, t, t, plane)
}

def createTileObject = {ROI roi, PathClass cls=null, MeasurementList mlist=null ->
    PathObjects.createTileObject(roi, cls, mlist)
}


def tileToDict(t, x, y) {
    def tileDict = [:]
    def boundsDict = [:]

    tileDict.put("tileId", [x, y])
    boundsDict.put("XY", [x * t, y * t])
    boundsDict.put("WH", [t, t])
    tileDict.put("tileBounds", boundsDict)

    return tileDict
}



def annotationToDict(t, a, filename="") {
    def result = [:]

    def clsName = a.getPathClass().name
    def roiBounds = getROIBounds(a.getROI())
    def tiles = getTileBoundsOverlap(t, *roiBounds[0], *roiBounds[1])

    result.put("filename", filename)
    result.put("class", clsName)
    result.put("tileSize", t)
    result.put("roiBounds", ["XY": roiBounds[0], "WH": roiBounds[1]])
    result.put("tiles", tiles.collect({tileToDict(t, *it)}))

    return result
}




def viewer = getCurrentViewer()
def hier = viewer.hierarchy
def plane = viewer.imagePlane

def rootObject = hier.getRootObject()
def annotations = hier.getAnnotationObjects()
def annotationsByType = pathObjectMap(annotations)[1]
def annotationTypes = annotationsByType.collect({k, v -> k})


def writeJson = {
    def filename = viewer.server.metadata.name
    def tileSize = 1024
    def results = annotations.collect({annotationToDict(tileSize, it, filename)})
    def gson = GsonTools.getInstance()
    try (Writer fh = new FileWriter("/Users/gennadiryan/Documents/gladstone/files/results/results.json", true)) {
        gson.toJson(results, fh)
    }
}

def tileSize = 1024
def tiles = pathObjectsTileMap(tileSize, annotations)

def tileClass = PathClassFactory.getPathClass("Tile")
def tileClassesByClass = tiles.groupBy({k, v -> v.collect({it.pathClass.name}).toSet()}).collectEntries({[it.key, it.value.keySet()]})
def tileClasses = tileClassesByClass.keySet().collectEntries({[it, [it.join(","), labelsToRGB(it)]]}).collectEntries({k, v -> [k, PathClassFactory.getDerivedPathClass(tileClass, *v)]})
def tileClassesByTile = [:]
for (entry in tileClassesByClass) {
    for (tile in entry.value) {
        tileClassesByTile.put(tile, tileClasses[entry.key])
    }
}


def tilePlane = ImagePlane.getPlane(0, 0)
def tileAnnotations = tileClassesByTile.collectEntries({k, v -> [k, createTileObject(createTileROI(tileSize, *k, tilePlane), v)]})
tileAnnotations = tileAnnotations.collect({k, v -> v})
hierarchy.addPathObjects(tileAnnotations)

//def tileROIs = tileClassesByClass.collectEntries(k, v -> [k, createTileROI()])

//tileClassesByClass.each{println it}




def labelsToRGB(labels) {
    def names = ["Core", "Diffuse", "Neuritic"]
    return getColorRGB(*(names.collect({it in labels ? 255 : 0})))
}






//def tileClasses = annotationTypes.collect({PathClassFactory.getDerivedPathClass(tileClass, it.name, 0)})
//def tileClassesBySize = tiles.groupBy({it.value.size()})
//def tileClassesByClass = tiles.groupBy({k, v -> v.collect({it.pathClass.name}).toSet()})


//colorMap.collectEntries({[it.key, it.value[0]]})


//def pathObjectsClassSet = {List<PathObject> objects -> objects.collect({it.pathClass.name}).toSet()}







//tileClasses.keySet()
//tileClasses[6]

//def clsDiffuse = annotationTypes.findAll({k -> k.name == "Diffuse"})[0]
//def annsDiffuse = annotationsByType.get(clsDiffuse)




//pathObjectsTileMap(1024, annotationsByType["Diffuse"])
//annotationsByType["Diffuse"]

//def pathObjectsToTiles(Integer t, List<PathROIObject> objects) {
////    return
//    def r = objects.collect({it.getROI()})
////    println r
//    println (getROIBounds(r[0]))
//    r = r.collect({getROIBounds(it)})
//    r = r.collect({getTileBoundsOverlap(t, *it[0], *it[1])})
//}



//


//annToDict(tileSize, anns[0])

////def results = []
//for (annotation in anns) {
//    def result = [:]
//
//    def label = annotation.pathClass.name
//    def roi = annotation.getROI()
//    def roiBounds = getROIBounds(roi)
//    def tiles = getTileBoundsOverlap(tileSize, *roiBounds[0], *roiBounds[1])
//
//    result.put("filename", "")
//    result.put("tileSize", tileSize)
//    result.put("roiBounds", ["XY": roiBounds[0], "WH": roiBounds[1]])
//    result.put("tiles", tiles.collect({tileToDict(tileSize, *it)}))
//    println result
//    println annotationToDict(
//}
//

//annotationToDict(1024, anns[0])
//anns[0]

//def (t, annotation) {
//    def result = [:]
//
//    def label = annotation.pathClass.name
//    ROI roi = annotation.getROI()
//    def roiBounds = getROIBounds(roi)
//    def tiles = getTileBoundsOverlap(t, *roiBounds[0], *roiBounds[1])
//
//    result.put("filename", "")
//    result.put("tileSize", t)
//    result.put("roiBounds", ["XY": roiBounds[0], "WH": roiBounds[1]])
//    result.put("tiles", tiles.collect({tileToDict(t, *it)}))
//
//    return result
//}

//def _getROIBounds(roi) {
//    map = [:]
//    map.put({round(roi.properties.get("bounds${it}"), floor=true)}, ["X", "Y"])
//    map.put({round(roi.properties.get("bounds${it}"), floor=false)}, ["Width", "Height"])
//    return mapOverLists(map)
//}
//

//
//def annToDict(Integer t, PathAnnotationObject annotation) {
//    def result = [:]
//    def regionBounds = [:]
//    def tileBounds = [:]
//
//    result.put("label", annotation.pathClass.name)
//    print(getROIBounds(annotations.ROI))
////
////    def roiBounds = getROIBounds(ann.getROI())
////    regionBounds.put("XY", roiBounds[0])
////    regionBounds.put("WH", roiBounds[1])
////    result.put("regionBounds", regionBounds)
////
////    result.put("tiles", getTileBoundsOverlap(t, *roiBounds[0], *roiBounds[1]).collect({tileToDict(t, *it)}))
//
//    println result
//}





//hier.removePathObjects(tiles)

//def annMapped = pathObjectMap(anns)
//def annTypes = annMapped[1].collect({k, v -> k})
//def tileSize = 1024
//def annTiles = annTypes.collect({pathObjectsToTiles(tileSize, annMapped[1][it])})//.each{println it}
//
//println annTypes[0].properties
//
//
//tileROI = createTileROI(tileSize, *annTiles[0][0], plane)
//tileObject = createTileObject(tileROI)
//tileObject.properties
//

//def fstAnns = annMapped[1][annTypes[0]].collect({it.getROI()}).collect({getROIBounds(it)}).collect({[it[0][0], it[0][1], it[1][0], it[1][1]]}).collect({getTileBoundsOverlap(1024, *it)})
//def rois = fstAnns
//fstAnns.size()
//def roisBounds = fstAnns
//roisBounds.size()
//def roisTiles = fstAnns.collect({getTileBoundsOverlap(1024, *it[0], *it[1])})
//roisTiles.size()

//pathObjectsToTiles(1024, annMapped[1][annTypes[0]]).each{println it}

//println rois

//def results = pathObjectsToTiles(1024, fstAnns)

//annType = annTypes[0]
//
//rois = pathObjectsToROIs(annMapped[1][annType])
//roisBounds = getROIsBounds(rois)
//getTileBoundsOverlap(1024, *roisBounds[0])
//
////pathObjectsToTiles(1024, annMapped[1][annType])

//annTypes.each{println it.properties}

//println annType.getColor() // -16711681
//annType.setColor(getColorRGB(100, 100, 0))
//println annType.getColor()
//annType.setColor(-16711681)

//tileClass = hier.tileObjects[0].pathClass
//newClass = getDerivedPathClass(tileClass, "derived", getColorRGB(100,100,100))
//println tileClass.properties
//println newClass.properties



//res = pathObjectsToROIs(objs)
//objs.collect({getROIBounds(it)})
//bounds = ROIsToBounds(res)
//Integer.methods.each{println it}
