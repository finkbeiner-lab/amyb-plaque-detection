import javafx.stage.FileChooser

import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.ImageRegion
import qupath.lib.io.GsonTools
import qupath.lib.objects.classes.PathClassTools


//TODO:
// general cleanup
// fix (or check) image server pixel size, plane
// find grid specs, methods to define custom overlays/grid overlays/text overlays

def tileSize = 1024

def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

String path = server.getPath()
filename = path.split("/")[-1]
filename = filename.split(".mrxs")[0]



// Get pair containing: list of unlabeled annotation ROIs, dictionary of per-label lists of annotation ROIs
def annotationsToROIsMap(annotations) {
    def unlabeled = []
    def labeled = [:]
    
    for (annotation in annotations) {
        label = annotation.pathClass
        roi = annotation.ROI
        if (label == null) {
            unlabeled.add(roi)
        } else {
            label = PathClassTools.splitNames(label)
            if (labeled[label] == null) {
                labeled.put(label, [])
            }
            
            labeled[label].add(roi)
        }
    }
    
    return [unlabeled, labeled]
}

// Get x, y axis tile overlaps from ROI coords represented as (x, y, w, h)
def coordsToTiles(t, x, y, w, h) {
    int x1 = x.intdiv(t)
    int y1 = y.intdiv(t)
    int x2 = (x + w).intdiv(t)
    int y2 = (y + h).intdiv(t)
    
    def tiles = []
    int xi = x1
    while (xi <= x2) {
        int yi = y1
        while (yi <= y2) {
            tiles.add([xi, yi])
            yi += 1
        }
        xi += 1
    }
    
    return tiles
}

// Get ROI's bounds in x, y axis
def ROIToCoords(roi) { //
    coords = []
    bounds = ["boundsX", "boundsY", "boundsWidth", "boundsHeight"]
    for (bound in bounds) {
        coords.add(roi.properties.get(bound).toInteger())
    }
    return coords
}

def ROIToPoints(roi) { //
    points = []
    dims = ["x", "y"]
    for (point in roi.allPoints) {
        pointCoords = []
        for (dim in dims) {
            pointCoords.add(point.properties.get(dim))
        }
        points.add(pointCoords)
    }
    return points
}

// Generate list of (t, x, y) tiles covered by the ROI's bounds
def ROIToTiles(t, roi) { //
    return coordsToTiles(t, *ROIToCoords(roi))
}


def ROIToTileDict(t, roi) { //
    def tiles = []
    for (tile in ROIToTiles(t, roi)) {
        tiles.add(tileToDict(t, *tile))
    }
    return tiles
}


// Generate dictionary of ROI keys, (t, x, y) tile values (as a list)
def ROIsToTilesMap(t, rois) { //
    def allROIs = [:]
    for (roi in rois) {
        allROIs.put(roi, ROIToTiles(t, roi))
    }
    return allROIs
}


def ROIToDict(t, roi) { //
    def roiDict = [:]
    def bounds = ROIToCoords(roi)
    def points = ROIToPoints(roi)
    
    boundsDict = [:]
    boundsDict.put("XY", [bounds[0], bounds[1]])
    boundsDict.put("WH", [bounds[2], bounds[3]])
    roiDict.put("roiBounds", boundsDict)
    roiDict.put("points", points)
    roiDict.put("tiles", ROIToTileDict(t, roi))
    
    
    return roiDict
}


def tileToDict(t, x, y) {
    tileDict = [:]
    z= 0
    
    //*********** Plot rectangel ROI ****************
//    def plane = ImagePlane.getPlane(z, 0)
//    roi = tileToROI(t, x, y, plane)
//    def annotation = PathObjects.createAnnotationObject(roi)
//    addObject(annotation)
    
    
    tileDict.put("tileId", [x, y])
    boundsDict = [:]
    boundsDict.put("XY", [x * t, y * t])
    boundsDict.put("WH", [t, t])
    tileDict.put("tileBounds", boundsDict)
    
    return tileDict
}



def viewer = qupath.lib.gui.scripting.QPEx.getCurrentViewer()
def hierarchy = viewer.hierarchy

def annotations = hierarchy.annotationObjects
def (unlabeled, labeled) = annotationsToROIsMap(annotations)


def roiTypes = ["Core", "Neuritic", "Diffuse", "CAA"]
def roisByType = [:]
roiTypes.each({
    for (k in labeled.keySet()) {
        if (k.get(0) == it) {
            if (!(k.get(0) in roisByType.keySet()))
                roisByType[k.get(0)] = []
            roisByType.get(k.get(0)).addAll(labeled.get(k))
        }
    }
})

def plane = viewer.imagePlane
def gson = GsonTools.getInstance(true)

def results = []
def roi_count = 0

for (item in roisByType) { 
    def roiType = item.key
    def rois = item.value
    def temp_results = [:]  
    temp_results["label"] = roiType
    temp_results["filename"] = filename
    
    def temp_attributes = []
    
    for (_item in ROIsToTilesMap(tileSize, rois)) {
        def roi = _item.key
        def tiles = _item.value
        temp_attributes.add(ROIToDict(tileSize, roi))
//        println ROIToDict(tileSize, roi)
    }
    
    temp_results["region_attributes"] =  temp_attributes
    
    results.add(temp_results)
    
    roi_count = roi_count + 1
}

//def out = gson.toJson(results)
Platform.runLater({
def newFile = (new FileChooser()).showSaveDialog()
gson.toJson(results, new FileWriter(newFile))
})

//savepath = "/mnt/new-nas/work/data/npsad_data/vivek/amy-def-mfg-jsons/" + filename + ".json"
//print(savepath)
//
//try (Writer writer = new FileWriter(savepath)) {
//        gson.toJson(results, writer);
//    }
