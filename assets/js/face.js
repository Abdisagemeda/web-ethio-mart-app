// import * as faceapi from './face-api.min.js';
Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('../assets/quantization/'),//loadFromDisk
  faceapi.nets.faceLandmark68Net.loadFromUri('../assets/quantization/'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('../assets/quantization/')
]);

// face detection
export async function start(file) {
  let image
  if (image) image.remove()
  image = await faceapi.bufferToImage(file);
  const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
  return detections;
}

export function load(){
  return true;
}
