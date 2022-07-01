import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClassFactory
import groovy.json.JsonSlurper
 
def imageData = QPEx.getCurrentImageData()
def server = imageData.getServer()

String path = server.getPath()
filename = path.split("/")[-1]
filename = filename.split(".mrxs")[0]


def roiToTiles(t, x, y, w, h) {
    x1 = x.intdiv(t)
    y1 = y.intdiv(t)
    x2 = (x + w).intdiv(t)
    y2 = (y + h).intdiv(t)
  
//    print(x)
//    print(y)
//    print(x1)
//    print(y1)
//    print(x2)
//    print(y2)
//     print(w)
//    print(h)
    
    tiles = []
    x = x1
    while (x <= x2) {
        y = y1
        while (y <= y2) {
            tiles.add([x, y])
            y += 1
        }
        x += 1
    }
    
    return tiles
}


def viewer = qupath.lib.gui.scripting.QPEx.getCurrentViewer()
def annotations = viewer.hierarchy.annotationObjects
//print(annotations.getData())
//
boolean prettyPrint = true
def gson1 = GsonTools.getInstance(prettyPrint)
x = gson1.toJson(annotations)

for (ele in x.items()) {
print(ele)

}
print(x)
//print(annotations.pathClass)


tileSize = 1024
def results = [:]
def gson = GsonTools.getInstance(true)

for (annotation in annotations) {
        label = annotation.pathClass
        roi = annotation.ROI
        allTiles = []
        for (roi in rois) {
            coords_dict = [:]
            coords = [roi.boundsX.toInteger(), roi.boundsY.toInteger(), roi.boundsWidth.toInteger(), roi.boundsHeight.toInteger()]
        
            roiTiles = roiToTiles(tileSize, *coords)
            results['fileName'] = filename
            results['tileSize'] = tileSize
            coords_dict['x'] = coords[0]
            coords_dict['y'] = coords[1]
            coords_dict['w'] = coords[2]
            coords_dict['h'] = coords[3]
            results['tileCoords'] = coords_dict
            print(annotation)
            
            results['label'] = label
            
            for (t in roiTiles) {
                    if (!(t in allTiles)) {
                        allTiles.add(t)
                    }
                } 
        }
  }

//print(allTiles)
 



def coordstoROI(coords, tileSize){
    
    int z = 0
    int t = 0

    def plane = ImagePlane.getPlane(z, t)
    PathClassFactory.getPathClass("Positive", ColorTools.packRGB(0, 0, 255))
    def roi = ROIs.createRectangleROI(coords[0] * tileSize, coords[1] * tileSize, tileSize, tileSize, plane)
    
   
   
    def annotation = PathObjects.createAnnotationObject(roi)
    addObject(annotation)

}



//for (tile in allTiles) {
//    coordstoROI(tile, tileSize)
//    results['tileSize'] = tileSize
//    results['tileCoords'] = coords
//
//}

// Write it into Json file format

try (Writer writer = new FileWriter("test.json")) {
        gson.toJson(results, writer);
    }



