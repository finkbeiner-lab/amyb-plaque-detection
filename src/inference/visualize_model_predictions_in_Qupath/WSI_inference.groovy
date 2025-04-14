import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.RectangleROI
import qupath.lib.regions.RegionRequest
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import org.json.JSONArray
import org.json.JSONObject
import qupath.lib.roi.PolygonROI
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.geom.Point2
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.dialogs.Dialogs
import qupath.lib.io.GsonTools


// ==========================================
// 🔧 CONFIGURATION SECTION
// ==========================================

// ✨ SIGN UP to get your own RunPod API Key and Endpoint
// 👉 https://runpod.io?ref=1mh0obxo

// ✅ STEP 1: Replace with your unique RunPod endpoint ID (from your model dashboard)
def url_endpoint = "your_endpoint_id_here"

// ✅ STEP 2: Replace with the full API URL for your endpoint
def apiUrl = "https://api.runpod.ai/v2/your_endpoint_id_here/run"

// ✅ STEP 3: Paste your personal RunPod API key here
def apiKey = "your_api_key_here"

// ✅ STEP 4: Set the full path where you want to save the output CSV
def csv_path = "/your/output/path/detections.csv"

def printMessage(String message) {
    println message
}

println "\uD83D\uDC40 Script started at " + new Date()

def imageData = getCurrentImageData()
def annotations = getAnnotationObjects()

def squareAnnotation = annotations.find { it.getROI() instanceof RectangleROI }

if (squareAnnotation == null) {
    printMessage("⚠️ No square annotation found.")
    return
}

def roi = squareAnnotation.getROI()
def xStart = roi.getBoundsX().toInteger()
def yStart = roi.getBoundsY().toInteger()
def width = roi.getBoundsWidth().toInteger()
def height = roi.getBoundsHeight().toInteger()

def tileSize = 1024
def server = imageData.getServer()

def allDetections = []
def labelCounters = [:].withDefault { 0 }

def client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build()

for (int tileY = yStart; tileY < yStart + height; tileY += tileSize) {
    for (int tileX = xStart; tileX < xStart + width; tileX += tileSize) {
        int tileWidth = Math.min(tileSize, xStart + width - tileX)
        int tileHeight = Math.min(tileSize, yStart + height - tileY)

        def regionRequest = RegionRequest.createInstance(server.getPath(), 1.0, tileX, tileY, tileWidth, tileHeight)
        def croppedImage = server.readRegion(regionRequest)

        def baos = new ByteArrayOutputStream()
        ImageIO.write(croppedImage, "png", baos)
        def imageBytes = baos.toByteArray()
        def encodedImage = imageBytes.encodeBase64().toString()

        def payload = """
        {
          "input": {
            "x": ${tileX},
            "y": ${tileY},
            "Image_buffer": "${encodedImage}"
          }
        }
        """

        def httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer ${apiKey}")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()

        def response = client.send(httpRequest, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode() == 200) {
            def responseBody = response.body()
            def jobIdMatcher = (responseBody =~ /"id"\s*:\s*"([^"]+)"/)
            def jobId = jobIdMatcher ? jobIdMatcher[0][1] : null

            if (jobId) {
                def statusUrl = "https://api.runpod.ai/v2/${url_endpoint}/status/${jobId}"
                def maxAttempts = 10
                def attemptInterval = 3000

                for (int i = 0; i < maxAttempts; i++) {
                    Thread.sleep(attemptInterval)

                    def statusRequest = HttpRequest.newBuilder()
                        .uri(URI.create(statusUrl))
                        .header("Authorization", "Bearer ${apiKey}")
                        .GET()
                        .build()

                    def statusResponse = client.send(statusRequest, HttpResponse.BodyHandlers.ofString())

                    if (statusResponse.statusCode() == 200) {
                        def statusBody = statusResponse.body()
                        def statusJson = new JSONObject(statusBody)
                        def status = statusJson.getString("status")

                        if (status == "COMPLETED") {
                            def output = statusJson.getString("output")
                            def jsonArray = new JSONArray(output)

                            jsonArray.each { result ->
                                def jsonObjectResult = result as JSONObject
                                def label = jsonObjectResult.getString('label')

                                labelCounters[label] += 1
                                def annotationId = "${label}-${labelCounters[label]}"

                                def annotX1 = jsonObjectResult.getInt('qupath_coord_x1')
                                def annotY1 = jsonObjectResult.getInt('qupath_coord_y1')
                                def annotX2 = jsonObjectResult.getInt('qupath_coord_x2')
                                def annotY2 = jsonObjectResult.getInt('qupath_coord_y2')
                                def annotRoi = new RectangleROI(annotX1, annotY1, annotX2 - annotX1, annotY2 - annotY1)

                                def annotation = new PathAnnotationObject(annotRoi)
                                def pathClass = getPathClass(label)
                                annotation.setPathClass(pathClass)
                                annotation.setName(annotationId)
                                annotation.getProperties().put("ID", annotationId)

                                annotation.getMeasurementList().putMeasurement("model_confidence", jsonObjectResult.getDouble("confidence"))
                                annotation.getMeasurementList().putMeasurement("brown_pixels", jsonObjectResult.getInt("brown_pixels"))
                                annotation.getMeasurementList().putMeasurement("area", jsonObjectResult.getDouble("area"))
                                annotation.getMeasurementList().putMeasurement("equivalent_diameter", jsonObjectResult.getDouble("equivalent_diameter"))
                                annotation.getMeasurementList().putMeasurement("eccentricity", jsonObjectResult.getDouble("eccentricity"))

                                def centroidList = jsonObjectResult.getJSONArray("centroid").toList()
                                annotation.getProperties().put("centroid", centroidList.toString())
                                annotation.getProperties().put("qupath_bbox", "(${annotX1}, ${annotY1}) to (${annotX2}, ${annotY2})")
                                annotation.getProperties().put("image_name", jsonObjectResult.getString("image_name"))

                                addObject(annotation)
                                allDetections << annotation

                                if (jsonObjectResult.has("polygon_coordinates")) {
                                    def polygonArray = jsonObjectResult.getJSONArray("polygon_coordinates")
                                    for (int j = 0; j < polygonArray.length(); j++) {
                                        def coords = polygonArray.getJSONArray(j)
                                        def points = []
                                        for (int k = 0; k < coords.length(); k++) {
                                            def point = coords.getJSONArray(k)
                                            def x = point.getDouble(0)
                                            def y = point.getDouble(1)
                                            points << new Point2(x, y)
                                        }
                                        def polygonROI = ROIs.createPolygonROI(points, null)
                                        def annotationPoly = PathObjects.createAnnotationObject(polygonROI, pathClass)
                                        annotationPoly.setName(annotationId)
                                        annotationPoly.getProperties().put("ID", annotationId)
                                        addObject(annotationPoly)
                                    }
                                }
                            }
                            break
                        } else if (status == "FAILED") {
                            def errorMatcher = (statusBody =~ /"error"\s*:\s*"([^"]+)"/)
                            def error = errorMatcher ? errorMatcher[0][1] : null
                            printMessage("❌ Job failed. Error: ${error}")
                            break
                        }
                    }
                }
            }
        }
    }
}

// Export summary to CSV
def csvHeader = "ID,QuPath Annotation ID,Label, Model Confidence,Brown Pixels,Area,Diameter,Eccentricity"
def csvRows = allDetections.collect { det ->
    def id = det.getProperties().get("ID")
    def name = det.getName()  // This is like "Diffuse-1"
    def label = det.getPathClass()?.getName()
    def conf = det.getMeasurementList().getMeasurementValue("model_confidence")
    def brown = det.getMeasurementList().getMeasurementValue("brown_pixels")
    def area = det.getMeasurementList().getMeasurementValue("area")
    def diam = det.getMeasurementList().getMeasurementValue("equivalent_diameter")
    def ecc = det.getMeasurementList().getMeasurementValue("eccentricity")
    return "${id},${name},${label},${conf},${brown},${area},${diam},${ecc}"
}

def outputFile = new File(csv_path)
outputFile.getParentFile().mkdirs()
outputFile.text = csvHeader + "\n" + csvRows.join("\n")

printMessage("\uD83D\uDCC4 Exported detection summary to: ${outputFile}")
printMessage("✅ Script finished at " + new Date())
