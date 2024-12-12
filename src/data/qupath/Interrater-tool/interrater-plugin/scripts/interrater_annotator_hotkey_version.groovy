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
import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClassFactory
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.tools.GuiTools
import javafx.scene.control.Label;
import javafx.scene.paint.Color;
import javafx.scene.input.KeyCode;
import qupath.lib.common.ColorTools
import java.util.ArrayList;


/// For adding classes and Keys, chnage just here 
// Define class and Keys
def all_classes = QPEx.getQuPath().availablePathClasses


//class_names =  ["Cored", "Diffuse", "Coarse-Grained", "Cotton-Wool", "Burned-Out","Unlabeled"]
//key_names = ["Ctrl+C", "Ctrl+F",  "Ctrl+G", "Ctrl+W",  "Ctrl+B","Ctrl+D"]
//color_list = [ [0, 255, 255],[255, 153, 102],[255, 0, 239],[160,170,90],[124,125,255],[255,0,0]]


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
                //dialog.setDialogPane(ParamDialog.this.dialogPane)
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
    String key
    String assignclass
    Integer color
    def AnnotatorDialog(PathObjectHierarchy hierarchy, String key, String assignclass, Integer color) {
        this.hierarchy = hierarchy
        this.key = key
        this.assignclass = assignclass
        this.color = color
        LinkedHashMap<String, PathClass> pathClassMap = new LinkedHashMap<String, PathClass>()
        pathClassMap.put("", PathClassFactory.getPathClassUnclassified())
        this.params = new ArrayList<Param<ChoiceParam<Object>, Object>>()
        params.add(new ChoiceParam<PathClass>("class", "Classification: ") {
            @Override void build() {
                super.build()
                this.setItems(pathClassMap)
            }
        })

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
       def imageData = getCurrentImageData()
       def name = getCurrentImageName()
       def cuurentHierarchy = imageData.getHierarchy()
       PathClass nopathClass = PathClassFactory.getPathClass("Unlabeled", makeRGB(255,255,255))
       if (this.assignclass=="DELETE-Annotation") {
           PathClass pathClass = PathClassFactory.getPathClass(this.assignclass)
           if (object!=null && object.isLocked()==false){
               clearSelectedObjects(true)
               def annotations = imageData.getHierarchy().getAnnotationObjects().findAll { it.getROI().getRoiName()!="Rectangle" && it.isLocked()==false}
               for (int i = 0; i < annotations.size(); i++) {
                    if(annotations[i].getName()==null) {
                        //PathClass nopathClass = PathClassFactory.getPathClass("Unlabeled", makeRGB(255,255,255))
                        annotations[i].setName((annotations.size()).toString()+":"+nopathClass)
                    }       
                }
               annotations = annotations.toSorted{it.getName().tokenize( ':' )[0].toInteger()}
               for (int i = 0; i < annotations.size(); i++) {
                 if(annotations[i].getPathClass()==null) {
                     annotations[i].setName(i.toString()+":"+"Unlabeled")
                 }
                 else{
                      annotations[i].setName(i.toString()+":"+annotations[i].getPathClass())
                      }
                }
               resolveHierarchy()
               def overlayOptions = getCurrentViewer().getOverlayOptions()
               overlayOptions.setShowNames(true)
               def allannotations = cuurentHierarchy.getAnnotationObjects()
               def path = PathPrefs.userPathProperty().get()+"/exports/" +  name + ".geojson"
               exportObjectsToGeoJson(allannotations,path,"PRETTY_JSON","FEATURE_COLLECTION")
           }
           return pathClass
       }
      else{  
        if(object!=null && object.isLocked()==false){
        PathClass pathClass = PathClassFactory.getPathClass(this.assignclass)
        if (pathClass != null) {
            object.setPathClass(pathClass)
            object.setColor(this.color)
        }
       def pointroi =  object.getROI()
       def x = pointroi.getBoundsX()
       def y = pointroi.getBoundsY()
       def size = 20
       //def imageData = getCurrentImageData()
       //def name = getCurrentImageName()
       //def cuurentHierarchy = imageData.getHierarchy()
       //def annotations = imageData.getHierarchy().getAnnotationObjects()
       def annotations = imageData.getHierarchy().getAnnotationObjects().findAll { it.getROI().getRoiName()!="Rectangle" && it.isLocked()==false}
        for (int i = 0; i < annotations.size(); i++) {
            if(annotations[i].getName()==null) {
                //PathClass nopathClass = PathClassFactory.getPathClass("Unlabeled", makeRGB(255,255,255))
                annotations[i].setName((annotations.size()-1).toString()+":"+nopathClass)
            }       
        }
       annotations = annotations.toSorted{it.getName().tokenize( ':' )[0].toInteger()}
       
       for (int i = 0; i < annotations.size(); i++) {
                 if(annotations[i].getPathClass()==null) {
                     annotations[i].setName(i.toString()+":"+nopathClass)
                 }
                 else{
                      annotations[i].setName(i.toString()+":"+annotations[i].getPathClass())
                      }
        }
       //int itr = annotations.size();
       //object.setName(itr +":"+ pathClass.toString())
       resolveHierarchy()
       def overlayOptions = getCurrentViewer().getOverlayOptions()
       overlayOptions.setShowNames(true)
       def allannotations = cuurentHierarchy.getAnnotationObjects()
       def path = PathPrefs.userPathProperty().get()+"/exports/" +  name + ".geojson"
       exportObjectsToGeoJson(allannotations,path,"PRETTY_JSON","FEATURE_COLLECTION")
       return pathClass
       }
       }
    }
    
    
    

    void buildAnnotatorCallable() {
        this.annotatorCallable = new Callable<PathClass>() {
            @Override PathClass call() {
               PathObjectSelectionModel selectionModel = AnnotatorDialog.this.hierarchy.getSelectionModel()             
               PathObject selected = selectionModel.getSelectedObject()
                  PathClass result = AnnotatorDialog.this.setObjectClass(selected)
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



def build(String keyCombination=null, String assignclass="", Integer color = "-1") {
    def gui = QPEx.getQuPath().getInstance()
    def hier = QPEx.getCurrentHierarchy()
    def app = new AnnotatorDialog(hier, keyCombination, assignclass, color)
    def menu = gui.installCommand(install_command_msg, app)
    print gui.viewer
    if (keyCombination != null) {
        //if (key_names.contains(keyCombination)) {
                 Platform.runLater( {
                    def scene = getQuPath().getStage().getScene()
                    scene.getAccelerators().put(KeyCombination.keyCombination(keyCombination), app);
                 })
       // }
     }
    return menu
}


install_command_msg = all_classes.toString()
all_classes.each { class_val ->
if (class_val.getColor()!=null){
    def class_name =  class_val.toString().tokenize(":")[0]
    print class_name
    def key_name =  class_val.toString().tokenize(":")[1]
    print key_name
    def color = class_val.getColor()
    print color
   def menu = build(key_name,class_name, color)
   }
}


