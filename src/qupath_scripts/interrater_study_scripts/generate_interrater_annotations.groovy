//import static qupath.lib.gui.scripting.QPEx.*
import groovy.io.FileType
import javafx.application.Platform
import qupath.lib.projects.Project;
import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.regions.ImageRegion
import qupath.lib.io.GsonTools
import qupath.lib.objects.classes.PathClassTools

File folder = new File('/Users/mahirwar/Downloads/interrater-test-jsons/') //assign json path

def tileSize = 1024

def project = getProject()
 //for (entry in project.getImageList()){
def image_list = project.getImageList()[0] // changing index for image_list
def image_data = image_list.readImageData()
describe(image_data)
 
PathObjectHierarchy heirarchy = image_data.getHierarchy()
Collection annotations = heirarchy.getAnnotationObjects()


path = image_list.getImageName()
filename = path.split("/")[-1]
filename = filename.split(".svs")[0]


def annotationsMap(annotations) {
    def unlabeled = []
    def labeled = [:]
    for (annotation in annotations) {
        label = annotation.pathClass
        if (label == null) {
            unlabeled.add(annotation)
        } else {
            label = PathClassTools.splitNames(label)
            if (labeled[label] == null) {
                labeled.put(label, [])
            }
            labeled[label].add(annotation)
        }
    }
    
    return [unlabeled, labeled]
}

// Get pair containing: list of unlabeled annotation ROIs, dictionary of per-label lists of annotation ROIs
def annotationsToROIsMap(annotations) {
    def unlabeled = []
    def labeled = [:]
    for (annotation in annotations) {
        print(annotation)
        label = annotation.pathClass
        //print(label)
        roi = annotation.ROI
        //print(roi)
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
    print(labeled)
    
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
def ROIToCoords(roi) {
    coords = []
    bounds = ["boundsX", "boundsY", "boundsWidth", "boundsHeight"]
    for (bound in bounds) {
        coords.add(roi.properties.get(bound).toInteger())
    }
    return coords
}

def ROIToPoints(roi) {
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
def ROIToTiles(t, roi) {
    return coordsToTiles(t, *ROIToCoords(roi))
}

def ROIToTileROIs(t, roi, plane) {
    def tileROIs = []
    for (tile in ROIToTiles(t, roi)) {
        tileROIs.add(tileToROI(t, *tile, plane))
    }
    return tileROIs
}

def ROIToTileDict(t, roi) {
    def tiles = []
    for (tile in ROIToTiles(t, roi)) {
        tiles.add(tileToDict(t, *tile))
    }
    return tiles
}

// Generate list of (t, x, y) tiles, one per ROI
def ROIsToTiles(t, rois) {
    def allTiles = []
    for (roi in rois) {
        allTiles.add(ROIToTiles(t, roi))
    }
    return allTiles
}

def ROIsToTileROIs(t, rois, plane) {
    def allTileROIs = []
    for (roi in rois) {
        allTileROIs.add(ROIToTileROIs(t, roi, plane))
    }
    return allTileROIs
}


// Generate dictionary of ROI keys, (t, x, y) tile values (as a list)
def ROIsToTilesMap(t, rois) {
    def allROIs = [:]
    for (roi in rois) {
        allROIs.put(roi, ROIToTiles(t, roi))
    }
    return allROIs
}

def ROIsToTileROIsMap(t, rois, plane) {
    def allTileROIs = [:]
    for (roi in rois) {
        allTileROIs.put(roi, ROIToTileROIs(t, roi, plane))
    }
    return allTileROIs
}

// Generate dictionary of (t, x, y) tile keys, ROI values
def ROIsToTilesInverseMap(t, rois) {
    def allTiles = [:]
    for (roiTile in ROIToTiles(t, roi)) {
        if (allTiles[roiTile] == null) {
            allTiles.put(roiTile, [roi])
        } else {
            allTiles[roiTile].add(roi)
        }
    }
    return allTiles
}



def ROIToDict(t, roi) {
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


// Generate ROI(s)/Annotations given (t, x, y) tile(s)
def tileToROI(t, x, y, plane) {
print(t)
print(x)
print(y)
    return qupath.lib.roi.ROIs.createRectangleROI(x * t, y * t, t, t, plane)
}

def tileToDict(t, x, y) {
    tileDict = [:]
    z= 0
    tileDict.put("tileId", [x, y])
    boundsDict = [:]
    boundsDict.put("XY", [x * t, y * t])
    boundsDict.put("WH", [t, t])
    tileDict.put("tileBounds", boundsDict)
    
    return tileDict
}

def tileToAnnotation(t, x, y, plane) {
    return qupath.lib.objects.PathObjects.createAnnotationObject(tileToROI(t, x, y, plane))
}

def tilesToAnnotations(t, tiles, plane) {
    def annotations = []
    for (tile in tiles) {
        annotations.add(tileToAnnotation(t, *tile, plane))
    }
    return annotations
}


def (unlabeled, labeled) = annotationsToROIsMap(annotations)

def gson = GsonTools.getInstance(true)
def results = []
def roi_count = 0
// since each rater is represented by a different roi, we get this info
for (item in annotations) { 
    def roiType = item.getPathClass()
    def rois = item.ROI
    def roiName = item.getROI().getRoiName()
    def objectName = item.getName()
    def roiArea = item.getROI().getArea()
    print(roiName)
    print(roiArea)
    print(objectName)
    def temp_results = [:]  
    temp_results["label"] = roiType
    temp_results["object_name"] = objectName
    temp_results["roiName"] = roiName
    if(roiName=="Polygon" && roiArea==4750){
        temp_results["raterName"] = "Osama"
       }
    else if(roiName=="Rectangle"){
        temp_results["raterName"] = "Harry"
       } 
    else if(roiName=="Polygon"){
        temp_results["raterName"] = "Max"
       }
     
    else if(roiName=="Ellipse"){
        temp_results["raterName"] = "Brittany"
       }   

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

def tile_list_dict = [:];


for (item in results) {
    key = item["region_attributes"][0]["tiles"][0]["tileId"]
    
      if (tile_list_dict[key] == null) {
           tile_list_dict.put(key, [])
        }
          
        tile_list_dict[key].add(item)
    
}


print(tile_list_dict)
savepath = "/Users/mahirwar/Downloads/interrater-study2-jsons/" + filename + ".json"
print(savepath)

try (Writer writer = new FileWriter(savepath)) {
        gson.toJson(tile_list_dict, writer);
    }