// This code assign ROI to points objects in Qupath file. We create different ROI for each rater, to visualize their annotations in Qupath
// and export these roi to match objects with predictions 

import qupath.lib.roi.ROIs
import qupath.lib.roi.EllipseROI;
import qupath.lib.objects.PathDetectionObject
import qupath.lib.geom.Point2
import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane

import qupath.lib.objects.PathObjects
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.RoiTools
import qupath.lib.geom.Point2
import qupath.lib.roi.EllipseROI
import qupath.lib.gui.prefs.PathPrefs

// define color code to classes
def pathClassMaps = [
        "Cored": [0, 255, 255],
        "Diffuse": [255, 153, 102],
        "CAA: CTRL+A": [150, 79, 239],
        "Coarse-Grained":[255, 0, 239],
        "Cotton-Wool":[160,170,90],
        "Burned-Out":[124,125,255],
    ]

//read through project
def project = getProject()

for (entry in project.getImageList()){
    def imageData = entry.readImageData()
    //def imageData = getCurrentImageData() # use for same image
    def hierarchy = imageData.getHierarchy()
    def annotations = hierarchy.getAnnotationObjects()
    print annotations[0].getROI().getRoiName()
    def pointAnnotations = annotations.findAll { it.getROI().getRoiName() == 'Points' }
    print annotations

    for (object in pointAnnotations){
        // get point roi
       def pointroi =  object.getROI()
       print pointroi
       def x = pointroi.getBoundsX()
       def y = pointroi.getBoundsY()
       def size = 40
       def name = entry.getImageName() //getCurrentImageName()
       // For Vivek, uncomment this
       //def roi = ROIs.createEllipseROI(x-size/2,y-size/2,size,size, ImagePlane.getDefaultPlane())
       // For Ceren, uncomment this
       //def roi = ROIs.createRectangleROI(x-size/2, y-size/2,size, size, ImagePlane.getDefaultPlane())
       // For max, uncomment this
       double[] xCoords = [x+0.0, x+50.0, x+100.0, x+75.0, x+25.0] 
       double[] yCoords = [y+0.0, y-20.0, y+0.0, y+50.0, y+50.0]
       def roi = ROIs.createPolygonROI(xCoords, yCoords, ImagePlane.getDefaultPlane())
       // This code remains same for all
       // this create annotation object with ROI and pathClass
       def annotationExpansion = PathObjects.createAnnotationObject(roi,object.getPathClass())
       print(object.getPathClass())
       print(object.getName())
       // Assigning color to the plaque class
       annotationExpansion.setColor(pathClassMaps[object.getPathClass()])
       // Assigning name to the annotation
       annotationExpansion.setName(pathClassMaps[object.getName()])
       annotationExpansion.setName(object.getName())
       // add annotation
       addObject(annotationExpansion)
    }
    // remove point objects
    removeObjects(pointAnnotations,true)
}