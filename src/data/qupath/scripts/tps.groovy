import groovy.transform.InheritConstructors

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;

import javafx.application.Platform;
import javafx.beans.InvalidationListener;
import javafx.beans.Observable;
import javafx.beans.binding.Bindings;
import javafx.beans.property.BooleanProperty;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.StringProperty;
import javafx.beans.property.FloatProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.SimpleFloatProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableStringValue;
import javafx.beans.value.ObservableValue;
import javafx.collections.ListChangeListener;
import javafx.collections.ObservableList;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.EventHandler;
import javafx.geometry.HPos;
import javafx.geometry.Pos;
import javafx.geometry.VPos;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ContextMenu;
import javafx.scene.control.Dialog;
import javafx.scene.control.DialogPane;
import javafx.scene.control.Label;
import javafx.scene.control.Menu;
import javafx.scene.control.RadioMenuItem;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ContextMenuEvent;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.stage.Modality
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Callback
import javafx.util.StringConverter

import org.controlsfx.control.action.Action;
import org.controlsfx.control.action.ActionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import qupath.ext.imagecombinerwarpy.gui.ImageCombinerWarpyServerOverlay;
import qupath.ext.imagecombinerwarpy.gui.ServerOverlay;
import qupath.ext.imagecombinerwarpy.gui.TPSTransform;
import qupath.ext.imagecombinerwarpy.gui.TPSTransformServer;
import qupath.ext.imagecombinerwarpy.gui.AffineTransformServer;

import qupath.lib.awt.common.AwtTools;
import qupath.lib.awt.common.BufferedImageTools;
import qupath.lib.color.ColorToolsAwt;
import qupath.lib.common.ColorTools;
import qupath.lib.common.ThreadTools;
import qupath.lib.display.ChannelDisplayInfo;
import qupath.lib.display.ImageDisplay;
import qupath.lib.gui.ActionTools;
import qupath.lib.gui.images.stores.AbstractImageRenderer;
import qupath.lib.gui.images.stores.DefaultImageRegionStore;
import qupath.lib.gui.images.stores.ImageRegionStoreFactory;
import qupath.lib.gui.images.stores.ImageRenderer;
import qupath.lib.gui.prefs.PathPrefs;
import qupath.lib.gui.scripting.QPEx;
import qupath.lib.gui.viewer.QuPathViewer;
import qupath.lib.gui.viewer.overlays.PathOverlay;
import qupath.lib.images.servers.AbstractImageServer
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerBuilder.ServerBuilder;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.io.GsonTools;
import qupath.lib.objects.PathObjects
import qupath.lib.objects.hierarchy.PathObjectHierarchy
import qupath.lib.projects.Project
import qupath.lib.projects.ProjectImageEntry
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.LineROI
import qupath.lib.roi.PointsROI


class PointUtils {
    static def pointsToArray(List points) {
        assert points.findAll({it.size() == 2}).size() == points.size()
        def pointsArray = new double[2][points.size()]
        points.withIndex().each({point, idx ->
            pointsArray[0][idx] = point[0]
            pointsArray[1][idx] = point[1]
        })
        return pointsArray
    }

    static def pointPairsToArrays(List pointPairs) {
        assert pointPairs.findAll({it.size() == 2}).size() == pointPairs.size()
        def srcPoints = pointPairs.collect({it.get(0)})
        def dstPoints = pointPairs.collect({it.get(1)})

        return [srcPoints, dstPoints].collect({pointsToArray(it)})
    }

    static def pointsFromHierarchy(PathObjectHierarchy hier) {
        return hier.getAnnotationObjects()
            .findAll({it.getROI() instanceof PointsROI && it.getROI().getNumPoints() == 1})
            .groupBy({it.getName()})
            .findAll({it.key != null && it.key.isInteger() && it.key.toInteger() > 0 && it.value.size() == 1})
            .collectEntries({[it.key.toInteger(), it.value.get(0)]})
            .collectEntries({[it.key, it.value.getROI().getAllPoints().get(0)]})
            .collectEntries({[it.key, [it.value.getX(), it.value.getY()]]})
    }

    static def pointArraysFromHierarchies(PathObjectHierarchy hier1, PathObjectHierarchy hier2) {
        def (points1, points2) = [hier1, hier2].collect({pointsFromHierarchy(it)})
        def keys = points1.keySet().findAll({it in points2.keySet()})
        return pointPairsToArrays(keys.collect({[points1.get(it), points2.get(it)]}).toList())
    }

    static def pointsAsHierarchy(PathObjectHierarchy hier1, PathObjectHierarchy hier2) {
        def (points1, points2) = [hier1, hier2].collect({def hier ->
            return hier.getAnnotationObjects()
                .findAll({it.getROI() instanceof PointsROI && it.getROI().getNumPoints() == 1})
                .groupBy({it.getName()})
                .findAll({it.key != null && it.key.isInteger() && it.key.toInteger() > 0 && it.value.size() == 1})
                .collectEntries({[it.key.toInteger(), it.value.get(0)]})
        })
        def hier = new PathObjectHierarchy()
        hier.addPathObjects(points1
            .findAll({it.key in points2.keySet()})
            .collect({
                def obj = PathObjects.createAnnotationObject(it.value.getROI())
                obj.setName(it.key.toString())
                return obj
            })
        )
        return hier
    }

    static def labelUnlabeledPoints(PathObjectHierarchy hier) {
        def grouped = hier.getAnnotationObjects()
            .findAll({it.getROI() instanceof PointsROI})
            .groupBy({it.getName()})
        if (null in grouped.keySet()) {
            def maxKey = grouped.keySet()
                .findAll({it != null && it.isInteger() && it.toInteger() > 0})
                .collect({it.toInteger()})
                .max()
            maxKey = 1 + ((maxKey == null) ? 0 : maxKey)
            grouped.get(null).withIndex().each({it.get(0).setName((it.get(1) + maxKey).toString())})
        }
    }
}


class Params {
    def paramDict
    def buttons

    def Params(Collection<ButtonType> buttons) {
        this.paramDict = [:]
        this.buttons = buttons
    }

    def add(String key, String label, Object param) {
        this.paramDict.put(key, [new Label(label), param])
    }

    def get(String key) {
        return this.paramDict.get(key)[1]
    }

    def buildGridPane() {
        def grid = new GridPane()
        grid.setHgap(10)
        grid.setVgap(10)
        this.paramDict.eachWithIndex({it, index -> grid.addRow(index, *it.value)})
        return grid
    }

    def buildPane() {
        def pane = new DialogPane()
        pane.setContent(this.buildGridPane())
        if (this.buttons != null)
            pane.getButtonTypes().setAll(this.buttons)
        return pane
    }

    static Callable<ButtonType> getAlertCallable(Stage owner, String title, String text) {
        Alert alert = new Alert(AlertType.INFORMATION)
        alert.initOwner(owner)
        alert.initModality(Modality.NONE)

        alert.setTitle(title)
        alert.setContentText(text)

        return new Callable<ButtonType>() {
            @Override ButtonType call() {
                def resp = alert.showAndWait()
                return resp.isPresent() ? null : resp.get()
            }
        }
    }

    static Callable<Integer> getDialogCallable(Stage owner, String title, DialogPane pane) {
        Dialog<ButtonType> dialog = new Dialog<>()
        dialog.initOwner(owner)
        dialog.initModality(Modality.NONE)

        dialog.setTitle(title)
        dialog.setDialogPane(pane)

        return new Callable<Integer>() {
            @Override Integer call() {
                Optional<ButtonType> resp = dialog.showAndWait()
                return resp.isPresent() ? dialog.getDialogPane().getButtonTypes().indexOf(resp.get()) : -1
            }
        }
    }
}


class ImageEntryBox extends ChoiceBox<ProjectImageEntry> {
    class ImageEntryConverter extends StringConverter<ProjectImageEntry> {
        Project project

        def ImageEntryConverter(Project project) {
            this.project = project
        }

        @Override String toString(ProjectImageEntry entry) {
            return entry == null ? "" : entry.getImageName()
        }

        @Override ProjectImageEntry fromString(String name) {
            def entries = this.project.getImageList().findAll({it.getImageName() == name})
            return entries.size() == 0 ? null : entries.get(0)
        }
    }

    def ImageEntryBox(Project project) {
        this(project, project.getImageList())
    }

    def ImageEntryBox(Project project, Collection<ProjectImageEntry> entries) {
        super()
        this.getItems().setAll(entries)
        this.setConverter(new ImageEntryConverter(project))
    }
}


class TPSTool implements Runnable {
    QuPathGUI gui

    def TPSTool(QuPathGUI gui) {
        this.gui = gui
    }

    static Params getParams(Project project, Collection<ProjectImageEntry> entries) {
        def params = new Params([ButtonType.CANCEL, ButtonType.APPLY, new ButtonType("Register points")])

        params.add("image1", "Base image: ", new ImageEntryBox(project, entries))
        params.add("image2", "Transform image: ", new ImageEntryBox(project, entries))

        params.add("stiffness", "Stiffness: ", new Slider(0, 1, 0))

        params.add("opacity", "Overlay opacity: ", new Slider(0, 1, 1))

        params.add("overlay", "Overlay transform: ", new CheckBox())
        params.get("overlay").setSelected(true)

        params.add("hierarchy", "Overlay hierarchy: ", new CheckBox())
        params.get("hierarchy").setSelected(true)

        return params
    }

    static TPSTransformServer getTransformServer(ImageData image1, ImageData image2, double stiffness=0.0) {
        def transform = new TPSTransform(*PointUtils.pointArraysFromHierarchies(image2.getHierarchy(), image1.getHierarchy()), stiffness)
        def transformServer = new TPSTransformServer(image2.getServer(), transform, ImageRegion.createInstance(0, 0, image1.getServer().getWidth(), image1.getServer().getHeight(), 0, 0))
        return transformServer
    }

    static ImageData putImageWithOverlay(QuPathGUI gui, QuPathViewer viewer, Project project, ImageServer baseImage, ImageServer overlayImage, PathObjectHierarchy hierarchy, DoubleProperty opacityProperty) {
        def image = new ImageData(*(hierarchy == null ? [baseImage] : [baseImage, hierarchy]))
        def overlay = new ServerOverlay(viewer, overlayImage)
        overlay.isOpacityProperty.set(true)
        overlay.opacityProperty.bind(opacityProperty)
        TPSTool.setImageWithOverlay(gui, viewer, project, image, overlay)

        // def prevImage = viewer.getImageData()
        // def prevEntry = prevImage == null ? null : project.getEntry(prevImage)
        // viewer.setImageData(image)
        // viewer.getCustomOverlayLayers().setAll(overlay)
        // if (prevEntry != null)
        //     project.removeImage(prevEntry, true)
        // gui.projectBrowser.refreshProject()
        return image
    }

    static void setImageWithOverlay(QuPathGUI gui, QuPathViewer viewer, Project project, ImageData image, PathOverlay overlay) {
        def prevImage = viewer.getImageData()
        def prevEntry = prevImage == null ? null : project.getEntry(prevImage)
        viewer.setImageData(image)
        viewer.getCustomOverlayLayers().setAll(overlay == null ? [] : overlay)
        if (prevEntry != null)
            project.removeImage(prevEntry, true)
        gui.projectBrowser.refreshProject()
    }

    static TPSTool getInstance(QuPathGUI gui) {
        return new TPSTool(gui)
    }

    void run() {
        def viewer = this.gui.getViewer()
        def owner = viewer.getView().getScene().getWindow()
        if (viewer.hasServer()) {
            Params.getAlertCallable(owner, "", "Main viewer is not empty").call()
            return
        }

        def project = this.gui.getProject()
        def images = this.gui.getViewers()
            .findAll({it != viewer})
            .findAll({it.hasServer()})
            .collect({it.getImageData()})
            .collectEntries({[project.getEntry(it), it]})

        def params = TPSTool.getParams(project, images.collect({it.key}))
        def dialog = Params.getDialogCallable(owner, "TPS Transform", params.buildPane())
        def res = dialog.call()
        while (res != 0) {
            def (entry1, entry2) = ["image1", "image2"].collect({params.get(it).getValue()})

            if (res == 1) {
                if (entry1 == null || entry2 == null) {
                    Params.getAlertCallable(owner, "", "Fewer than two images selected").call()
                } else {
                    def (image1, image2) = [entry1, entry2].collect({images.get(it)})
                    double stiffness = params.get("stiffness").getValue()
                    boolean overlay = params.get("overlay").isSelected()
                    boolean hierarchy = params.get("hierarchy").isSelected()

                    def servers = [image1.getServer(), TPSTool.getTransformServer(image1, image2, stiffness)]
                    def imageData = TPSTool.putImageWithOverlay(gui, viewer, project, *(overlay ? servers : servers.reverse()), hierarchy ? PointUtils.pointsAsHierarchy(*([image1, image2].collect({it.getHierarchy()}))) : null, params.get("opacity").valueProperty())
                }
            } else if (res == 2) {
                [entry1, entry2]
                    .findAll({it != null})
                    .collect({images.get(it)})
                    .toSet()
                    .each({PointUtils.labelUnlabeledPoints(it.getHierarchy())})
            }

            res = dialog.call()
        }
        TPSTool.setImageWithOverlay(gui, viewer, project, null, null)
    }
}


def gui = QPEx.getQuPath().getInstance()
gui.installCommand("TPS Tool", new TPSTool(gui))
