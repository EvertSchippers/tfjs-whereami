/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// Importing TensorFlow:
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

// Helper class to add training samples in memory.
import {ControllerDataset} from './controller_dataset';

// The button:
const button1 = document.getElementById('b1');
const button2 = document.getElementById('b2');
const button3 = document.getElementById('b3');
const trainbutton = document.getElementById('train');

// The events:
button1.addEventListener('mousedown', () => addSample(0));
button2.addEventListener('mousedown', () => addSample(1));
button3.addEventListener('mousedown', () => addSample(2));
trainbutton.addEventListener('mousedown', () => train());

// Display the probability:
function setprobability(label, probability)
{
  document.getElementById('thumb-' + (label + 1)).style.border = "thick solid " + toColor(probability); 
}

// Drawing a thumbnail:
export function drawThumb(img, label) {
    const thumbCanvas = document.getElementById('thumb-' + (label + 1));
    draw(img, thumbCanvas);

    var p = [0,0,0];
    p[label] = 1;

    setprobability(0,p[0]);
    setprobability(1,p[1]);
    setprobability(2,p[2]);
}

// Drawing on the live canvas:
export function draw(image, canvas) {
  const [width, height] = [224, 224];
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
    imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
    imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 3; //PSYCH!!

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  console.log(layer.outputShape);
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// Picks a frame from the webcam, computes embeddings and adds it as
// a training sample to the specified label.
async function addSample(label)
{
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // Draw the preview thumbnail.
  drawThumb(img, label);
  img.dispose();
}

/**
 * Sets up and trains the classifier.
 */
async function train() {

  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(0.0001);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainbutton.innerText = 'Loss: ' + logs.loss.toFixed(5);
      }
    }
  });

  // Funnily enough you'll notice it will start predicting already while training
  // as the "fit" call is async.
  predict();
}

let isPredicting = false;


async function predict() {
  isPredicting = true;
 
  while (isPredicting) {
    // Capture the frame from the webcam.
    const img = await getImage();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    const embeddings = truncatedMobileNet.predict(img);
    // 7x7x256 apparently


    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    const predictions = model.predict(embeddings);
    
    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    const predictedClass = await predictions.as1D().data();


    setprobability(0, predictedClass[0]);
    setprobability(1, predictedClass[1]);
    setprobability(2, predictedClass[2]);

    img.dispose();

    await tf.nextFrame();
  }
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}



async function init() {
 
  const cfg  =  {facingMode: 'environment' }
  webcam = await tfd.webcam(document.getElementById('webcam'), cfg);
 
  truncatedMobileNet = await loadTruncatedMobileNet();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
}



// Helpers to convert probability into color:
function dec2hex(dec) {
  return Number(parseInt( dec , 10)).toString(16);
}
function pad(h){ //adds leading 0 to single-digit codes
  if(h.length==1) return "0"+h;
  else return h;
}
function toColor(probability)
{
   var r = 255 * probability;
   return "#" + pad(dec2hex(0.2 * r)) + pad(dec2hex(r)) + pad(dec2hex(0.5 * r));
}



// Initialize the application.
init();
