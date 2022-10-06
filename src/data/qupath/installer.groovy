import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import java.net.URL

import javafx.application.Application
import javafx.application.Platform
import javafx.beans.value.ObservableValue
import javafx.collections.FXCollections
import javafx.collections.ListChangeListener
import javafx.collections.ObservableList
import javafx.concurrent.Task
import javafx.scene.control.Alert
import javafx.scene.control.Alert.AlertType
import javafx.scene.control.ButtonType
import javafx.scene.control.Dialog
import javafx.scene.control.DialogPane
import javafx.scene.control.CheckBox
import javafx.scene.control.ChoiceBox
import javafx.scene.control.Label
import javafx.scene.control.ListCell
import javafx.scene.control.ListView
import javafx.scene.control.SelectionMode
import javafx.scene.control.TextArea
import javafx.scene.control.TextField
import javafx.scene.layout.GridPane
import javafx.scene.layout.Region
import javafx.stage.Modality
import javafx.stage.FileChooser
import javafx.stage.DirectoryChooser
import javafx.util.Callback
import javafx.util.StringConverter

import java.util.concurrent.Callable
import java.util.concurrent.FutureTask



import qupath.lib.gui.scripting.QPEx
import qupath.lib.gui.prefs.PathPrefs


class Installer {
    static def getGroupedPaths(List lines, Map groups) {
        return lines
            .collect({Paths.get(it)})
            .findAll({it.getRoot() == null && it.getNameCount() > 1})
            .groupBy({it.getName(0).toString()})
            .findAll({groups.keySet().contains(it.key)})
            .collectEntries({key, value -> [key, value
                .collect({it.subpath(1, it.getNameCount())})
                .findAll({path -> groups.get(key).any({path.getName(path.getNameCount() - 1).toString().endsWith("." + it)})})
                .collect({it.toString()})]})
                
    }
    
    static def getGroupedPairs(Path localRoot, URL remoteRoot, Map paths) {
        return paths
            .collectEntries({[it.key, [[
                new File(localRoot.toFile(), it.key),
                new URL(remoteRoot, it.key + "/"),
            ], it.value]]})
            .collectEntries({key, value -> [key, value.get(1).collect({[
                it,
                new File(value.get(0).get(0), it),
                new URL(value.get(0).get(1), it)
            ]})]})
    }
    
    static def addPairs(List pairs) {
        def bothPairs = pairs.groupBy({it.get(1).isFile()})
        def (installPairs, updatePairs) = [false, true].collect({bothPairs.get(it, [])})
        
        def params = new Params()
        params.add("install", "Install: ", new FixedSizeListView(installPairs.collect({it.get(0)})))
        params.add("update", "Update: ", new FixedSizeListView(updatePairs.collect({it.get(0)})))
        
        def callable = FXUtils.dialogCallable(params.build())
        if (!callable.call())
            return null
        
        def addPairs = []
        ["install": installPairs, "update": updatePairs].each({k, v -> params.get(k).getSelectionModel().getSelectedIndices().toArray().each({addPairs.add(v.get(it))})})
        addPairs.each({
            if (!it.get(1).getParentFile().isDirectory())
                it.get(1).getParentFile().mkdirs()
            it.get(1).setBytes(it.get(2).getBytes())
        })
    }
    
    static def getManifestPairs() {
        def localRoot = Paths.get(PathPrefs.getUserPath()) // Local user directory
        def remoteRoot = new URL("https://raw.githubusercontent.com/finkbeiner-lab/amyb-plaque-detection/main/src/data/qupath/") // Equivalent remote directory
        def remoteManifest = new URL(remoteRoot, "manifest.txt") // Remote manifest file
        def manifestGroups = ["scripts": ["groovy"], "extensions": ["jar"]] // Filter manifest file according to these descriptors
        
        def manifestPaths = getGroupedPaths(remoteManifest.readLines(), manifestGroups)
        def manifestPairs = getGroupedPairs(localRoot, remoteRoot, manifestPaths)
        return manifestPairs
    }
    
    static def getInstaller(String groupName) {
        def pairs = getManifestPairs().get(groupName, [])
        
        return new Runnable() {
            @Override void run() {
                Installer.addPairs(pairs)
            }
        }
    }
}

Platform.runLater(Installer.getInstaller("scripts"))





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

    def build() {
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


class FixedSizeListView extends ListView<String> {
    def FixedSizeListView(List<String> items) {
        this(items, 24)
    }
    
    def FixedSizeListView(List<String> items, double fixedCellSize) {
        super()
        this.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE)
        this.setItems(FXCollections.observableArrayList(items))
        this.setFixedCellSize(fixedCellSize)
        this.setPrefHeight((this.getItems().size() * this.getFixedCellSize()) + 2)
    }
}


class OptionBox extends ChoiceBox<Integer> {
    class OptionConverter extends StringConverter<Integer> {
        List<String> options
        String defaultOption

        def OptionConverter(List<String> options, String defaultOption) {
            this.options = options
            this.defaultOption = defaultOption
        }

        @Override String toString(Integer index) {
            return index == null ? this.defaultOption : this.options.get(index)
        }

        @Override Integer fromString(String value) {
            return this.options.indexOf(value)
        }
    }

    def OptionBox(List<String> options) {
        this(options, new String())
    }

    def OptionBox(List<String> options, String defaultOption) {
        super()
        this.getItems().setAll((0 ..< options.size()).toArray())
        this.setConverter(new OptionConverter(options, defaultOption))
    }
}


class FXUtils {
    static def alertCallable(String msg) {
        return new Callable<Boolean>() {
            @Override Boolean call() {
                Alert alert = new Alert(AlertType.CONFIRMATION)
                alert.initModality(Modality.NONE)
                alert.setContentText(msg)
                Optional<ButtonType> resp = alert.showAndWait()
                return resp.isPresent() && resp.get() == ButtonType.OK
            }
        }
    }

    static def dialogCallable(DialogPane pane) {
        return new Callable<Boolean>() {
            @Override Boolean call() {
                Dialog dialog = new Dialog()
                dialog.initModality(Modality.NONE)
                dialog.setDialogPane(pane)
                Optional<ButtonType> resp = dialog.showAndWait()
                return resp.isPresent() && resp.get() == ButtonType.OK
            }
        }
    }
    
    static def listViewCallable(Map optLists) {
        def params = new Params()
        optLists.each({
            def lv = new ListView()
            lv.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE)
            lv.setItems(FXCollections.observableArrayList(it.value.get(1)))
            params.add(it.key, it.value.get(0), lv)
        })
        def callable = FXUtils.dialogCallable(params.build())
        return new Callable<Map>() {
            @Override Map call() {
                if (!callable.call())
                    return null
                return params.paramDict.collectEntries({[it.key, it.value.get(1).getSelectionModel().getSelectedIndices().toArray()]})
            }
        }
    }

    static def optionCallable(Map options) {
        def params = new Params()
        options.each({
            params.add(it.key, it.value.get(0), new OptionBox(it.value.get(1)))
        })
        def callable = FXUtils.dialogCallable(params.build())
        return new Callable<Map>() {
            @Override Map call() {
                if (!callable.call())
                    return null
                return params.paramDict.collectEntries({[it.key, it.value.get(1).getValue()]})
            }
        }

    }
}
