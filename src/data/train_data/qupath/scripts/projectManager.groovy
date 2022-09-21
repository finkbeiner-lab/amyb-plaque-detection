import java.io.*
import java.util.concurrent.Callable
import java.util.concurrent.FutureTask

import java.awt.image.BufferedImage

import javafx.application.Platform
import javafx.scene.control.Alert
import javafx.scene.control.Alert.AlertType
import javafx.scene.control.ButtonType
import javafx.scene.control.ChoiceBox
import javafx.scene.control.Dialog
import javafx.scene.control.DialogPane
import javafx.scene.control.Label
import javafx.scene.control.TextArea
import javafx.scene.control.TextField
import javafx.scene.layout.GridPane
import javafx.stage.DirectoryChooser
import javafx.stage.FileChooser
import javafx.stage.Modality
import javafx.util.StringConverter

import qupath.lib.gui.scripting.QPEx

import qupath.lib.gui.commands.ProjectImportImagesCommand
import qupath.lib.images.servers.ImageServer
import qupath.lib.images.servers.ImageServerBuilder
import qupath.lib.images.servers.ImageServerBuilder.DefaultImageServerBuilder
import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.images.servers.ImageServers
import qupath.lib.images.servers.ServerTools
import qupath.lib.projects.*

import qupath.lib.images.servers.openslide.OpenslideImageServer
import qupath.lib.images.servers.openslide.OpenslideServerBuilder


class ProjectManager {
    File projectFile
    Project project
    Class provider

    def ProjectManager(File projectDir) {
        this(projectDir, null, null)
    }

    def ProjectManager(File projectDir, String projectFileName) {
        this(projectDir, projectFileName, null)
    }

    def ProjectManager(File projectDir, Class provider) {
        this(projectDir, null, provider)
    }

    def ProjectManager(File projectDir, String projectFileName, Class provider) {
        assert projectDir.isDirectory()
        assert provider == null || provider in ProjectManager.getInstalledProviders()
        this.projectFile = new File(projectDir, (projectFileName != null ? projectFileName : ProjectIO.DEFAULT_PROJECT_NAME) + "." + ProjectIO.DEFAULT_PROJECT_EXTENSION)
        if (this.projectFile.isFile())
            this.project = ProjectIO.loadProject(this.projectFile, BufferedImage.class)
        else
            this.project = Projects.createProject(this.projectFile, BufferedImage.class)
        this.provider = provider
    }

    static def getInstalledProviders() {
        return ServerBuilders.getInstalledBuilders().collect({it.getClass()})
    }

    def createImageEntry(File imageFile) {
        return this.createImageEntry(imageFile, this.provider)
    }

    def createImageEntry(File imageFile, Class provider) {
        def builder = provider != null ? ServerBuilders.getBuilder(imageFile, provider) : null
        return builder != null ? this.project.addImage(builder) : null
    }

    static def initializeImageEntry(ProjectImageEntry entry) {
        try (def server = entry.getServerBuilder().build()) {
            entry.setImageName(ServerTools.getDisplayableImageName(server))
            entry.setThumbnail(ProjectImportImagesCommand.getThumbnailRGB(server, null))
            return entry
        } catch (Exception e) {
            return null
        }
    }

    def addImageEntries(List<File> imageFiles) {
        def removeEntries = []
        def entries = imageFiles.collect({this.createImageEntry(it)}).collect({
            if (it != null && ProjectManager.initializeImageEntry(it) == null) {
                removeEntries.add(it)
                return null
            } // otherwise "it" has been successfully initialized
            return it
        })
        this.project.removeAllImages(removeEntries, true)
        return entries
    }

    def addImageEntriesDialog() {
        def imageFiles = (new FileChooser()).showOpenMultipleDialog()
        if (imageFiles == null)
            return null
        return this.addImageEntries(imageFiles).withIndex().collect({[imageFiles.get(it.get(1)), it.get(0)]})
    }

    static def providerDialog() {
        def providers = ProjectManager.getInstalledProviders()
        def resp = FXUtils.optionCallable(["provider": ["Image server provider: ", providers.collect({it.getName()})]]).call()
        if (resp == null)
            return null
        def idx = resp.get("provider")
        if (idx == null || idx < 0 || idx >= providers.size())
            return null
        return providers.get(idx)
    }

    static def projectDialog() {
        def projectDir = (new DirectoryChooser()).showDialog()
        return projectDir != null ? new ProjectManager(projectDir) : null
    }

    static def projectsDialog() {
        def projectsDir = (new DirectoryChooser()).showDialog()
        if (projectsDir == null)
            return null
        def provider = ProjectManager.providerDialog()
        if (provider == null)
            return null

        def params = new Params()
        params.add("projectName", "Project name: ", new TextField())
        params.get("projectName").setText(new String())
        while (FXUtils.dialogCallable(params.build()).call()) {
            def projectName = params.get("projectName").getText()
            params.get("projectName").setText(new String())
            if (projectName == null || projectName.length() == 0) {
                if (FXUtils.alertCallable("No project name entered").call())
                    continue
                else
                    break
            }
            def projectDir = new File(projectsDir, projectName)
            if (!projectDir.isDirectory())
                projectDir.mkdirs()
            def project = new ProjectManager(projectDir, provider)

            def entries = project.addImageEntriesDialog()
            if (entries == null)
                continue
            def entryGroups = entries.groupBy({it.get(1) != null})
            def results = new Params()
            results.add("added", "Entries added: ", new TextArea())
            results.add("failed", "Entries failed: ", new TextArea())
            entryGroups.each({
                key, value ->
                    def sb = new StringBuilder()
                    value.collect({it.get(0)}).each({sb.append(it.toString() + "\n")})
                    results.get(key ? "added" : "failed").setText(sb.toString())
            })
            if (!FXUtils.dialogCallable(results.build()).call() && true in entryGroups.keySet()) {
                project.project.removeAllImages(entryGroups.get(true).collect({it.get(1)}), true)
            }
            project.project.syncChanges()
        }
    }

    static def projectsFromFoldersDialog() {}
}


class ServerBuilders {
    static def getInstalledBuilders() {
        return ImageServerProvider.getInstalledImageServerBuilders()
    }

    static def getImageSupport(File imageFile, ImageServerBuilder builder) {
        return builder.checkImageSupport(imageFile.toURI())
    }

    static def getAllImageSupports(File imageFile) {
        return ServerBuilders.getInstalledBuilders().collect({ServerBuilders.getImageSupport(imageFile, it)})
    }

    static def getBuilder(File imageFile, Class providerClass) {
        return DefaultImageServerBuilder.createInstance(providerClass, imageFile.toURI())
    }

    static def getBuilder(File imageFile, ImageServerBuilder builder) {
        return ServerBuilders.getBuilder(imageFile, builder.getClass())
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


Platform.runLater({
    // def proj = ProjectManager.projectDialog()
    // if (proj != null && proj.setProvider()) {
    //     def fileEntryPairs = proj.addImageEntries()
    //     if (fileEntryPairs != null) {
    //         def groups = fileEntryPairs.groupBy({it.get(1) != null})
    //         if (false in groups.keySet())
    //             FXUtils.alertCallable("Loading failed: " + groups.get(false).collect({it.get(0)}).toString()).call()
    //         if ((true in groups.keySet() && FXUtils.alertCallable("Loading succeeded: " + groups.get(true).collect({it.get(0)}).toString()).call()) || FXUtils.alertCallable("Nothing loaded; save project anyway?").call())
    //             proj.project.syncChanges()
    //     }
    // }

    ProjectManager.projectsDialog()
})
