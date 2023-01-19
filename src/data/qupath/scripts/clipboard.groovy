import groovy.transform.InheritConstructors

import java.io.BufferedReader
import java.io.FileReader
import java.io.File
import java.io.StringReader
import java.io.StringWriter

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
import javafx.scene.input.KeyCombination
import javafx.scene.layout.GridPane
import javafx.stage.Modality
import javafx.stage.FileChooser
import javafx.stage.DirectoryChooser
import javafx.util.StringConverter

import com.google.gson.stream.JsonReader
import com.google.gson.stream.JsonWriter

import org.json.*
import org.apache.commons.io.FileUtils

import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.QuPathGUI

import qupath.lib.geom.Point2

import qupath.lib.io.GsonTools
import qupath.lib.io.ROITypeAdapters

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



class Clipboard {
    QuPathGUI gui
    PathObjectHierarchy hier

    String clipboard


    def Clipboard(QuPathGUI gui) {
        this.gui = gui
    }

    def refresh() {
        def imageData = this.gui.getImageData()
        if (imageData == null)
            this.hier = null
        else
            this.hier = imageData.getHierarchy()
        return this.hier
    }


    def jsonArrToList(JSONArray jsonArr) {
        return jsonArr.length() == 0 ? [] : (0 ..< jsonArr.length()).toArray().collect({jsonArr.get(it)})
    }

    String serializeROI(ROI roi) {
        def writer = new StringWriter()
        ROITypeAdapters.ROI_ADAPTER_INSTANCE.write(new JsonWriter(writer), roi)
        return writer.toString()
    }

    ROI deserializeROI(String json) {
        return ROITypeAdapters.ROI_ADAPTER_INSTANCE.read(new JsonReader(new StringReader(json)))
    }

    JSONArray serialize(List<PathAnnotationObject> anns) {
        return (new JSONArray(anns.collect({
            def map = [:]
            map.put("roi", this.serializeROI(it.getROI()))
            map.put("pathClasses", it.getPathClass() == null ? [] : PathClassTools.splitNames(it.getPathClass()))
            if (it.getName() != null)
                map.put("name", it.getName())
            if (it.getColorRGB() != null)
                map.put("color", it.getColorRGB())
            return map
        })))
    }

    List deserialize(String json) {
        return jsonArrToList(new JSONArray(json)).collect({
            def map = [:]
            map.put("roi", deserializeROI(it.get("roi")))
            map.put("pathClasses", (ArrayList<String>) jsonArrToList(it.get("pathClasses")))
            map.put("name", !it.has("name") ? null : (String) it.get("name"))
            map.put("color", !it.has("color") ? null : (Integer) it.get("color"))
            return map
        }).collect({
            def obj = PathObjects.createAnnotationObject(it.get("roi"), it.get("pathClasses").size() == 0 ? PathClassFactory.getPathClassUnclassified() : PathClassFactory.getPathClass(it.get("pathClasses")))
            obj.setName(it.get("name"))
            obj.setColorRGB(it.get("color"))
            return obj
        })
    }


    def copy() {
        return new Runnable() {
            void run() {
                if (Clipboard.this.refresh() == null)
                    return
                def sel = Clipboard.this.hier.getSelectionModel()
                def anns = Clipboard.this.hier.getAnnotationObjects().findAll({sel.isSelected(it)})
                Clipboard.this.clipboard = Clipboard.this.serialize(anns)
            }
        }
    }

    def paste() {
        return new Runnable() {
            void run() {
                if (Clipboard.this.refresh() == null)
                    return
                Clipboard.this.hier.addPathObjects(Clipboard.this.deserialize(Clipboard.this.clipboard))
            }
        }
    }
    
    static Clipboard getInstance(QuPathGUI gui) {
        return new Clipboard(gui)
    }
}


return Clipboard
