import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RectangleROI
import qupath.lib.roi.PointsROI
import qupath.lib.roi.interfaces.ROI
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


class WSIInferenceExtension {
    // Define the API endpoint and API key
    def apiUrl = "https://api.runpod.ai/v2/8xa742wutj14fm/run"
    def apiKey = "api-key, for runpod cloud environment"

    // Method to execute the main functionality
    def execute() {
        // Get the current image data
        def imageData = getCurrentImageData()

        // Get the user-drawn square annotation
        def annotations = getAnnotationObjects()

        // Print the ROI type and class for each annotation
        annotations.each { annotation ->
            def roi = annotation.getROI()
            println "Annotation ROI Type: ${roi.getRoiType().toString()}"
            println "Annotation ROI Class: ${roi.getClass().getName()}"
        }

        def squareAnnotation = annotations.find { it.getROI() instanceof RectangleROI }

        if (squareAnnotation == null) {
            println "No square annotation found."
            return
        }

        def roi = squareAnnotation.getROI()
        def x1 = roi.getBoundsX().toInteger()
        def y1 = roi.getBoundsY().toInteger()
        def width = roi.getBoundsWidth().toInteger()
        def height = roi.getBoundsHeight().toInteger()

        // Crop the image based on the annotation coordinates
        def server = imageData.getServer()
        def regionRequest = RegionRequest.createInstance(server.getPath(), 1.0, x1, y1, width, height)
        def croppedImage = server.readRegion(regionRequest)

        // Convert the cropped image to bytes
        def baos = new ByteArrayOutputStream()
        ImageIO.write(croppedImage, "png", baos)
        def imageBytes = baos.toByteArray()

        // Encode image bytes to Base64
        def encodedImage = imageBytes.encodeBase64().toString()

        // Prepare the payload
        def payload = """
{
  "input": {
    "x": ${x1},
    "y": ${y1},
    "Image_buffer": "${encodedImage}"
  }
}
"""

        // Create an HTTP client
        def client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build()

        // Create the HTTP request to start the job
        def httpRequest = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer ${apiKey}")
            .POST(HttpRequest.BodyPublishers.ofString(payload))
            .build()

        // Send the request to start the job
        def response = client.send(httpRequest, HttpResponse.BodyHandlers.ofString())

        // Check if the job was started successfully
        if (response.statusCode() == 200) {
            def responseBody = response.body()
            def jobIdMatcher = (responseBody =~ /"id"\s*:\s*"([^"]+)"/)
            def jobId = jobIdMatcher ? jobIdMatcher[0][1] : null

            if (jobId) {
                println "Job started with ID: ${jobId}"

                // Poll for the job result
                def statusUrl = "https://api.runpod.ai/v2/8xa742wutj14fm/status/${jobId}"
                def maxAttempts = 10
                def attemptInterval = 3000 // 10 seconds

                for (int i = 0; i < maxAttempts; i++) {
                    Thread.sleep(attemptInterval)

                    def statusRequest = HttpRequest.newBuilder()
                        .uri(URI.create(statusUrl))
                        .header("Authorization", "Bearer ${apiKey}")
                        .GET()
                        .build()

                    def statusResponse = client.send(statusRequest, HttpResponse.BodyHandlers.ofString())
                    println statusResponse.statusCode()

                    if (statusResponse.statusCode() == 200) {
                        def statusBody = statusResponse.body()
                        def statusJson = new JSONObject(statusBody) // Parse statusBody as JSON
                        def status = statusJson.getString("status") // Get the "status" field from the JSON object

                        if (status == "COMPLETED") {
                            def output = statusJson.getString("output") // Get the "output" field from the JSON object
                            println "Job completed. Result: ${output}"
                            
                            if (output) {
                                // Parse the JSON output using org.json
                                def jsonArray = new JSONArray(output)

                                // Print details for each result
                                jsonArray.each { result ->
                                    def jsonObjectResult = result as JSONObject
                                    println "Image Name: ${jsonObjectResult.getString('image_name')}"
                                    println "Label: ${jsonObjectResult.getString('label')}"
                                    println "Confidence: ${jsonObjectResult.getDouble('confidence')}"
                                    println "Brown Pixels: ${jsonObjectResult.getInt('brown_pixels')}"
                                    println "Area: ${jsonObjectResult.getDouble('area')}"
                                    println "Equivalent Diameter: ${jsonObjectResult.getDouble('equivalent_diameter')}"
                                    println "Centroid: ${jsonObjectResult.getJSONArray('centroid').toList()}"
                                    println "Eccentricity: ${jsonObjectResult.getDouble('eccentricity')}"
                                    println "QuPath Coordinates: (${jsonObjectResult.getInt('qupath_coord_x1')}, ${jsonObjectResult.getInt('qupath_coord_y1')}) to (${jsonObjectResult.getInt('qupath_coord_x2')}, ${jsonObjectResult.getInt('qupath_coord_y2')})"
                                    println "-----------------------------"

                                    // Extract QuPath coordinates
                                    int annotX1 = jsonObjectResult.getInt('qupath_coord_x1')
                                    int annotY1 = jsonObjectResult.getInt('qupath_coord_y1')
                                    int annotX2 = jsonObjectResult.getInt('qupath_coord_x2')
                                    int annotY2 = jsonObjectResult.getInt('qupath_coord_y2')

                                    // Extract label
                                    String label = jsonObjectResult.getString('label')

                                    // Create a RectangleROI
                                    def annotRoi = new RectangleROI(annotX1, annotY1, annotX2 - annotX1, annotY2 - annotY1)

                                    // Create a PathAnnotationObject with the ROI
                                    def annotation = new PathAnnotationObject(annotRoi)

                                    // Set the classification (label) for the annotation
                                    annotation.setPathClass(getPathClass(label))
                                    annotation.setName(label)

                                    // Add the annotation to the current image
                                    addObject(annotation)

                               
                                }
                            }
                            break
                        } else if (status == "FAILED") {
                            def errorMatcher = (statusBody =~ /"error"\s*:\s*"([^"]+)"/)
                            def error = errorMatcher ? errorMatcher[0][1] : null
                            println "Job failed. Error: ${error}"
                            break
                        } else if (status == "IN_QUEUE") {
                            println "Job is in queue. Waiting..."
                        } else {
                            println "Job status: ${status}. Waiting..."
                        }
                    } else {
                        println "Error checking job status: ${statusResponse.statusCode()} - ${statusResponse.body()}"
                    }
                }
            } else {
                println "Error: Unable to retrieve Job ID from the response."
            }
        } else {
            println "Error starting job: ${response.statusCode()} - ${response.body()}"
        }
    }
}

// Register the extension in the extension tab
def extension = new WSIInferenceExtension()
extension.execute()






