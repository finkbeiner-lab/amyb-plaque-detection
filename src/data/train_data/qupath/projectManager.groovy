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
    List<Class> providers
    Class provider

    def ProjectManager(File projectDir) {
        this(projectDir, ProjectIO.DEFAULT_PROJECT_NAME)
    }

    def ProjectManager(File projectDir, String projectFileName) {
        assert projectDir.isDirectory() && projectFileName != null
        this.projectFile = new File(projectDir, projectFileName + "." + ProjectIO.DEFAULT_PROJECT_EXTENSION)
        if (this.projectFile.isFile())
            this.project = ProjectIO.loadProject(this.projectFile, BufferedImage.class)
        else
            this.project = Projects.createProject(this.projectFile, BufferedImage.class)

        this.providers = ServerBuilders.getInstalledBuilders().collect({it.getClass()})
        this.provider = null
    }

    def getProviders() {
        return this.providers
    }

    def setProvider() {
        def resp = FXUtils.optionCallable(["provider": ["Image server provider: ", this.getProviders().collect({it.getName()})]]).call()
        if (resp == null)
            return false
        def idx = resp.get("provider")
        if (idx == null || idx < 0 || idx >= this.providers.size())
            return false
        return this.setProvider(this.providers.get(idx))
    }

    def setProvider(Class provider) {
        this.provider = provider in this.providers ? provider : null
        return this.provider != null
    }



    def addImageEntries() {
        def imageFiles = (new FileChooser()).showOpenMultipleDialog()
        if (imageFiles == null)
            return null
        return this.addImageEntries(imageFiles).withIndex().collect({[imageFiles.get(it.get(1)), it.get(0)]})
    }

    def createImageEntry(File imageFile) {
        def builder = this.provider != null ? ServerBuilders.getBuilder(imageFile, this.provider) : null
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
        return imageFiles
            .collect({this.createImageEntry(it)})
            .collect({
                if (it == null || ProjectManager.initializeImageEntry(it) != null)
                    return it
                this.project.removeImage(it, true)
                return null
            })
    }

    static def projectDialog() {
        def projectDir = (new DirectoryChooser()).showDialog()
        return projectDir != null ? new ProjectManager(projectDir) : null
    }
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
        def callable = FXUtils.dialogCallable(params.pane())
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

  def pane() {
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
    def proj = ProjectManager.projectDialog()
    if (proj != null && proj.setProvider()) {
        def fileEntryPairs = proj.addImageEntries()
        if (fileEntryPairs != null) {
            def groups = fileEntryPairs.groupBy({it.get(1) != null})
            if (false in groups.keySet())
                FXUtils.alertCallable("Loading failed: " + groups.get(false).collect({it.get(0)}).toString()).call()
            if ((true in groups.keySet() && FXUtils.alertCallable("Loading succeeded: " + groups.get(true).collect({it.get(0)}).toString()).call()) || FXUtils.alertCallable("Nothing loaded; save project anyway?").call())
                proj.project.syncChanges()
        }
    }
})
