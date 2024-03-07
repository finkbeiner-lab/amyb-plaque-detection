import groovy.transform.InheritConstructors
import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.tools.GuiTools
import qupath.lib.objects.*
import qupath.lib.objects.classes.*
import qupath.lib.objects.hierarchy.*
import qupath.lib.objects.hierarchy.events.*

import javafx.application.Platform
import javafx.concurrent.Task
import javafx.scene.control.Alert
import javafx.scene.control.Alert.AlertType
import javafx.scene.control.ButtonType
import javafx.scene.control.Dialog
import javafx.scene.control.DialogPane
import javafx.scene.control.ChoiceBox
import javafx.scene.control.ListView;
import javafx.scene.control.ListCell;

import javafx.scene.control.Label
import javafx.scene.control.TextArea
import javafx.scene.control.TextField
import javafx.scene.input.KeyCombination
import javafx.scene.layout.GridPane
import javafx.util.Callback
import javafx.util.StringConverter

import java.util.concurrent.Callable
import java.util.concurrent.FutureTask

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


// Documented at src/data/train_data/qupath/annotatorApp.README.txt


abstract class Param<S, T> {
    abstract T param
    String paramName
    Label paramLabel

    def Param(String paramName, String paramLabel) {
        this.paramName = paramName
        this.paramLabel = new Label(paramLabel)
    }

    abstract void build()
    abstract S get()
    abstract void set(S value)
}

@InheritConstructors
class TextParam extends Param<String, TextField> {
    void build() { this.param = new TextField() }
    String get() { return this.param.getText() }
    void set(String value) { this.param.setText(value) }
    void setEditable(Boolean editable) { this.param.setEditable(editable) }
}

@InheritConstructors
class TextAreaParam extends Param<String, TextArea> {
    void build() { this.param = new TextArea() }
    String get() { return this.param.getText() }
    void set(String value) { this.param.setText(value) }
    void setEditable(Boolean editable) { this.param.setEditable(editable) }
}

@InheritConstructors
class ChoiceParam<T> extends Param<T, ListView<T>> {
    void build() { this.param = new ListView<T>() }
    //T get() { return this.param.getValue() }
    //void set(T value) { this.param.setValue(value) }
    //void setConverter(StringConverter<T> value) { this.param.setConverter(value) }
     T get() { return this.param.getSelectionModel().getSelectedItem(); }
    void set(T value) { this.param.getSelectionModel().select(value); }
    void setConverter(StringConverter<T> value) { 
        this.param.setCellFactory(listView -> new ListCell<T>() {
            @Override
            protected void updateItem(T item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty ? null : value.toString(item));
            }
        });
    }
    void setItems(List<T> values) { this.param.getItems().setAll(values); }
    //void setItems(List<T> values) { this.param.getItems().setAll(values) }
    void setItems(LinkedHashMap<String, T> values) {
        this.setConverter(new StringConverter<T>() {
            @Override
            public String toString(T object) {
                return values.entrySet().stream()
                        .filter(entry -> Objects.equals(entry.getValue(), object))
                        .map(Map.Entry::getKey)
                        .findFirst()
                        .orElse(null);
            }

            @Override
            public T fromString(String string) {
                return values.get(string);
            }
        });
        this.setItems(new ArrayList<>(values.values()));
    }
    //void setItems(LinkedHashMap<String, T> values) {
     //   this.setConverter(new StringConverter() {
     //       @Override String toString(T object) { return values.find({it.value == object})?.key }
     //       @Override T fromString(String string) { return values.get(string) }
     //   })
     //   this.setItems(values.collect({it.value}))
    //}
}


abstract class ParamDialog<P, R> implements Callable {
    // List<Param<P, Object>> params
    P params
    GridPane gridPane
    DialogPane dialogPane
    Callback<ButtonType, R> resultConverter
    Callable<Optional<R>> dialogCallable

    // def ParamDialog(List<Param<R, Object>> params) {
    def ParamDialog(P params) {
        this.params = params
    }

    def ParamDialog() {
        this.params = null
    }

    void buildParams() {
        this.params.each({it.build()})
    }

    void buildGridPane() {
        this.gridPane = new GridPane()
        this.gridPane.setHgap(10)
        this.gridPane.setVgap(10)
        this.params.eachWithIndex({it, index -> this.gridPane.addRow(index, it.paramLabel, it.param)})
    }

    void buildDialogPane() {
        this.dialogPane = new DialogPane()
        this.dialogPane.getButtonTypes().setAll(ButtonType.OK, ButtonType.CANCEL)
        this.dialogPane.setContent(this.gridPane)
    }

    // Not implemented
    abstract void buildResultConverter()

    void buildDialogCallable() {
        this.dialogCallable = new Callable<Optional<R>>() {
            @Override Optional<R> call() {
                Dialog<R> dialog = new Dialog<R>()
                dialog.setDialogPane(ParamDialog.this.dialogPane)
                dialog.setResultConverter(ParamDialog.this.resultConverter)
                return dialog.showAndWait()
            }
        }
    }

    // Not implemented
    abstract Callable<Optional<R>> build()

    Optional<R> call() {
        return GuiTools.callOnApplicationThread(this.build())
    }

    Optional<R> callAndWait() {
        FutureTask<Optional<R>> future = new FutureTask<Optional<R>>(this.build())
        Platform.runLater(future)
        return future.get()
    }
}


class AnnotatorDialog implements Runnable {
    ArrayList<Param<ChoiceParam<Object>, Object>> params
    Callable<PathClass> annotatorCallable
    Callable<Optional> objectOrdering
    ParamDialog<ArrayList<Param<ChoiceParam<Object>, Object>>, ArrayList<Object>> dialog
    def imageData = getCurrentImageData()
    def cuurentHierarchy = imageData.getHierarchy()
    PathObjectHierarchy hierarchy = cuurentHierarchy
    
    def AnnotatorDialog(PathObjectHierarchy hierarchy) {
        this.hierarchy = hierarchy
        LinkedHashMap<String, PathClass> pathClassMap = new LinkedHashMap<String, PathClass>()
        pathClassMap.put("", PathClassFactory.getPathClassUnclassified())
        ["Cored", "Diffuse", "Coarse-Grained", "Cotton-Wool", "Burned-Out"].each({pathClassMap.put(it, PathClassFactory.getPathClass(it))})

        //LinkedHashMap<String, Integer> intensityMap = new LinkedHashMap<String, Integer>()
        //intensityMap.put("0", 0)
        //[1, 2, 3].each({intensityMap.put(it.toString() + "+", it)})

        this.params = new ArrayList<Param<ChoiceParam<Object>, Object>>()
        params.add(new ChoiceParam<PathClass>("class", "Classification: ") {
            @Override void build() {
                super.build()
                this.setItems(pathClassMap)
            }
        })
        //params.add(new ChoiceParam<Integer>("intensity", "Confidence: ") {
        //    @Override void build() {
        //        super.build()
        //        this.setItems(intensityMap)
        //    }
        //})

        this.dialog = new ParamDialog<ArrayList<Param<ChoiceParam<Object>, Object>>, ArrayList<Object>>(params) {
            @Override void buildResultConverter() {
                this.resultConverter = new Callback<ButtonType, ArrayList<Object>>() {
                    @Override ArrayList<Object> call(ButtonType param) {
                        if (param == ButtonType.OK) {
                            return AnnotatorDialog.this.params.collect({it.get()})
                        }
                        return null
                    }
                }
            }

            @Override Callable<Optional<ArrayList<Object>>> build() {
                this.buildParams()
                this.buildResultConverter()
                this.buildGridPane()
                this.buildDialogPane()
                this.buildDialogCallable()
                return this.dialogCallable
            }
        }
    }

    Callable<Optional<ButtonType>> buildAlertCallable(String msg) {
        return new Callable<Optional<ButtonType>>() {
            @Override Optional<ButtonType> call() {
                Alert alert = new Alert(AlertType.INFORMATION)
                alert.setContentText(msg)
                return alert.showAndWait()
            }
        }
    }




    PathClass setObjectClass(PathObject object) {
        Optional<List<Object>> resultOpt = this.dialog.build().call()
        if (!resultOpt.isPresent()) {
            return null
        }

        List<Object> result = resultOpt.get()
        PathClass pathClass = result.get(0)
        //Integer intensity = result.get(1)
        
        print pathClass
        //print intensity
                
        
        if (pathClass == null || pathClass == PathClassFactory.getPathClassUnclassified()) {
            return null
        }
        //if (intensity == null || !(0 <= intensity && intensity <= 3)) {
        //    return null
        ///}

        //pathClass = this.getPathClass(pathClass, intensity)
        if (pathClass != null) {
            object.setPathClass(pathClass)
        }
        
       def pointroi =  object.getROI()
       print pointroi
       def x = pointroi.getBoundsX()
       def y = pointroi.getBoundsY()
       def size = 20
       def imageData = getCurrentImageData()
       def name = getCurrentImageName()
       def cuurentHierarchy = imageData.getHierarchy()
       def annotations = imageData.getHierarchy().getAnnotationObjects()
       def ellipticalROIs = annotations.findAll { annotation -> annotation instanceof PathAnnotationObject && annotation.getROI() != null && annotation.getName()!=null }
       int itr = ellipticalROIs.size() + 1;
       def roi = ROIs.createEllipseROI(x-size/2,y-size/2,size,size, ImagePlane.getDefaultPlane())
       def annotationExpansion = PathObjects.createAnnotationObject(roi,pathClass)
       annotationExpansion.setName(itr +":"+ pathClass.toString())
       addObject(annotationExpansion)
       resolveHierarchy()
       def overlayOptions = getCurrentViewer().getOverlayOptions()
       overlayOptions.setShowNames(true)
       def allannotations = cuurentHierarchy.getAnnotationObjects()
       def path = PathPrefs.userPathProperty().get()+"/exports/" +  name + ".geojson"
       //exportObjectsToGeoJson(allannotations,path, "FEATURE_COLLECTION") 
       exportObjectsToGeoJson(allannotations,path,"PRETTY_JSON","FEATURE_COLLECTION")
       return pathClass
    }
    

    void buildAnnotatorCallable() {
        this.annotatorCallable = new Callable<PathClass>() {
            @Override PathClass call() {
                PathObjectSelectionModel selectionModel = AnnotatorDialog.this.hierarchy.getSelectionModel()
                //AnnotatorDialog.this.objectOrdering.call()
                if (selectionModel.noSelection()) {
                    AnnotatorDialog.this.buildAlertCallable("No selected objects found").call()
                    return null
                }
                if (!selectionModel.singleSelection()) {
                    AnnotatorDialog.this.buildAlertCallable("More than one selected object found").call()
                    return null
                }

                PathObject selected = selectionModel.getSelectedObject()
                if (!selected.isAnnotation()) {
                    AnnotatorDialog.this.buildAlertCallable("Selected object is not an annotation").call()
                    return null
                }
                if (selected.isLocked()) {
                    AnnotatorDialog.this.buildAlertCallable("Selected object is locked").call()
                    return null
                }

                PathClass result = AnnotatorDialog.this.setObjectClass(selected)
                if (result == null) {
                    AnnotatorDialog.this.buildAlertCallable("No PathClass set").call()
                } else {
//                     AnnotatorDialog.this.buildAlertCallable("PathClass set to: " + result.toString()).call()
                }
                return result
            }
        }
    }

     







    Callable<PathClass> build() {
        this.buildAnnotatorCallable()
        return this.annotatorCallable
    }
    
    

    @Override void run() {
        this.build().call()
    }
}



def build(String keyCombination=null) {
    def gui = QPEx.getQuPath().getInstance()
    def hier = QPEx.getCurrentHierarchy()
    def app = new AnnotatorDialog(hier)
    def menu = gui.installCommand("annotatorApp: Ctrl+D", app)
    print gui.viewer
    if (keyCombination != null) {
        //def hier = QPEx.getCurrentHierarchy()
        //def app = new AnnotatorDialog(hier)
        //def menu = gui.installCommand("annotatorApp", app)
        Platform.runLater({
            //menu.acceleratorProperty().unbind()
            //menu.accelerator = KeyCombination.keyCombination(keyCombination)
            def scene = getQuPath().getStage().getScene()
            scene.getAccelerators().put(KeyCombination.keyCombination(keyCombination), app);
     })
     
     
     }
    return menu
}


def menu = build("Ctrl+D")







