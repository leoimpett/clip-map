{/* <script> */}
/////////////
//  STEP 1 //
/////////////

var workersInitialised = false; // will be set to true after workers are initialized


// var maxNImages = 6000;

// doAll function
function doAll() {
  searchBtn.disabled = true;
  // disable changeFolder
  // changeFolder.disabled = true;





  // initialize workers
  if (!workersInitialised){initializeWorkers();}
  

  // pick directory
    pickDirectory({source:'local'});
    console.log("directory picked")

}


// first we need to download the models and initialize the workers 
// window.MODEL_NAME = "clip_vit_32";
window.MODEL_NAME = "clip_vit_32_uint8"
window.modelData = {
  clip_vit_32: {
    image: {
      modelUrl: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-${quantized ? "uint8" : "float32"}.onnx`,
      embed: async function(blob, session) {
        let rgbData = await getRgbData(blob);
        const feeds = {input: new ort.Tensor('float32', rgbData, [1,3,224,224])};
        const results = await session.run(feeds);
        const embedVec = results["output"].data; // Float32Array
        return embedVec;
      }
    },
    text: {
      modelUrl: (quantized) => `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-${quantized ? "uint8" : "float32-int32"}.onnx`,
      embed: async function(text, session) {
        if(!window.textTokenizerClip) {
          let Tokenizer = (await import("https://deno.land/x/clip_bpe@v0.0.6/mod.js")).default;
          window.textTokenizerClip = new Tokenizer(); 
        }
        let textTokens = window.textTokenizerClip.encodeForCLIP(text);
        textTokens = Int32Array.from(textTokens);
        const feeds = {input: new ort.Tensor('int32', textTokens, [1, 77])};
        const results = await session.run(feeds);
        return [...results["output"].data];
      },
    }
  },
  lit_b16b: {
    image: {
      modelUrl: () => 'https://huggingface.co/rocca/lit-web/resolve/main/embed_images.onnx',
      embed: async function(blob, session) {
        
        // TODO: Maybe remove tf from this code so you can remove the whole tfjs dependency
        blob = await bicubicResizeAndCenterCrop(blob);
        let inputImg = new Image();
        await new Promise(r => inputImg.onload=r, inputImg.src=URL.createObjectURL(blob));
        let img = tf.browser.fromPixels(inputImg);
        img = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
        let float32RgbData = img.dataSync();
        
        const feeds = {'images': new ort.Tensor('float32', float32RgbData, [1,224,224,3])};
        const results = await session.run(feeds);
        return results["Identity_1:0"].data;
      },
    },
    text: {
      modelUrl: () => 'https://huggingface.co/rocca/lit-web/resolve/main/embed_text_tokens.onnx',
      embed: async function(text, session) {
        // Here we use a custom tokenizer that is not part of the model
        if(!window.bertTextTokenizerLit) {
          window.bertTextTokenizerLit = await import("./bert-text-tokenizer.js").then(m => new m.BertTokenizer());
          await window.bertTextTokenizerLit.load();
        }
        let textTokens = window.bertTextTokenizerLit.tokenize(text);
        textTokens.unshift(101); // manually put CLS token at the start
        textTokens.length = 16;
        textTokens = [...textTokens.slice(0, 16)].map(e => e == undefined ? 0 : e); // pad with zeros to length of 16
        textTokens = Int32Array.from(textTokens);
        const feeds = {'text_tokens': new ort.Tensor('int32', textTokens, [1,16])};
        const results = await session.run(feeds);
        return [...results["Identity_1:0"].data];
      }
    }
  },
};

let imageWorkers = [];
let onnxImageSessions = [];
let onnxTextSession;
let textTokenizer;

let imageResults; 

async function initializeWorkers() {

  workersInitialised = true; 

  console.log("initialising workers")


  // show downloadingProgressBars
  dwpgbar = document.getElementById("downloadingProgressBars")
  if(dwpgbar){
    dwpgbar.style.display = "block";
    console.log('showing progressbar')
  }

  // initWorkersBtn.disabled = true;
  // numThreadsEl.disabled = true;
  
  let useQuantizedModel = false;
  
  if(MODEL_NAME.endsWith("_uint8")) {
    MODEL_NAME = MODEL_NAME.replace(/_uint8$/g, "");
    useQuantizedModel = true;
  }
  
  let imageOnnxBlobPromise = downloadBlobWithProgress(window.modelData[MODEL_NAME].image.modelUrl(useQuantizedModel), function(e) {
    let ratio = e.loaded / e.total;
    // imageModelLoadingProgressBarEl.value = ratio;
    // imageModelLoadingMbEl.innerHTML = Math.round(ratio*e.total/1e6)+" MB";
  });

  let textOnnxBlobPromise = downloadBlobWithProgress(window.modelData[MODEL_NAME].text.modelUrl(useQuantizedModel), function(e) {
    let ratio = e.loaded / e.total;
    // textModelLoadingProgressBarEl.value = ratio;
    // textModelLoadingMbEl.innerHTML = Math.round(ratio*e.total/1e6)+" MB";
  });

  let [imageOnnxBlob, textOnnxBlob] = await Promise.all([imageOnnxBlobPromise, textOnnxBlobPromise])
  console.log("Blob sizes: ", imageOnnxBlob.size, textOnnxBlob.size);

  let imageModelUrl = window.URL.createObjectURL(imageOnnxBlob);
  let textModelUrl = window.URL.createObjectURL(textOnnxBlob);

  // console.log("URLs: ", imageModelUrl, textModelUrl);
  
  // let numImageWorkers = Number(numThreadsEl.value);
  let numImageWorkers = 4;
  
  // Inference latency is about 5x faster with wasm threads, but this requires these headers: https://web.dev/coop-coep/ I'm using this as a hack (in enable-threads.js) since Github pages doesn't allow setting headers: https://github.com/gzuidhof/coi-serviceworker
  if(self.crossOriginIsolated) {
    ort.env.wasm.numThreads = Math.ceil(navigator.hardwareConcurrency / numImageWorkers) / 2; // divide by two to utilise only half the CPU's threads because trying to use all the cpu's threads actually makes it slower
  }

  // workerInitProgressBarEl.max = numImageWorkers + 2; // +2 because of text model and bpe library
  
  let imageModelExecutionProviders = ["wasm"]; // webgl is not compatible with this model (need to tweak conversion data/op types)

  for(let i = 0; i < numImageWorkers; i++) {
    let session = await ort.InferenceSession.create(imageModelUrl, { executionProviders: imageModelExecutionProviders }); 
    onnxImageSessions.push(session);
    imageWorkers.push({
      session,
      busy: false,
    });
    // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;
  }
  console.log("Image model loaded.");

  onnxTextSession = await ort.InferenceSession.create(textModelUrl, { executionProviders: ["wasm"] }); // webgl is not compatible with this model (need to tweak conversion data/op types)
  console.log("Text model loaded.");
  // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;

  window.URL.revokeObjectURL(imageModelUrl);
  window.URL.revokeObjectURL(textModelUrl);

  window.vips = await Vips(); // for bicubicly resizing images (since that's what CLIP expects)
  window.vips.EMBIND_AUTOMATIC_DELETELATER = false;

  // workerInitProgressBarEl.value = Number(workerInitProgressBarEl.value) + 1;

  // disableCtn(initCtnEl);
  // enableCtn(pickDirCtnEl);

  // hide searchSpinner
  // document.getElementById("searchSpinner").style.display = "none";


  // hide downloadingProgressBars
  // document.getElementById("downloadingProgressBars").style.display = "none";
  console.log('hiding progressbar')

  


}


/////////////
//  STEP 2 //
/////////////
let directoryHandle;
let embeddingsFileHandle;
let embeddings;
let dataSource;
async function pickDirectory(opts={}) {
  dataSource = opts.source;
   
  if(dataSource === "local") {
    if(!window.showDirectoryPicker) return alert("Your browser does not support some modern features (specifically, File System Access API) required to use this web app. Please try updating your browser, or switching to Chrome, Edge, or Brave.");
    directoryHandle = await window.showDirectoryPicker();
    embeddingsFileHandle = await directoryHandle.getFileHandle(`${window.MODEL_NAME}_embeddings.tsv`, {create:true});
    

  }
  
  // let redditEmbeddingsBlob;
  // if(dataSource === "reddit") {
  //   if(window.MODEL_NAME !== "clip_vit_32") return alert("Sorry, there are only pre-computed Reddit image embeddings for the CLIP ViT-B/32 model at the moment.");
  //   if(!removeRedditNsfwEl.checked && !confirm("Are you sure you'd like to see NSFW Reddit images?")) return;
  //   if(removeRedditNsfwEl.checked) alert("Note that NSFW images are filtered from Reddit using CLIP, and CLIP can make mistakes, so some NSFW images may still be shown.");
      
  //   // pickDirectoryBtn.disabled = true;
  //   // useRedditImagesBtn.disabled = true;
  //   // useRedditImagesBtn.textContent = "Loading...";
  //   // redditLoadProgressCtn.style.display = "";
    
  //   redditEmbeddingsBlob = await downloadBlobWithProgress("https://huggingface.co/datasets/rocca/top-reddit-posts/resolve/main/clip_embeddings_top_50_images_per_subreddit.tsv.gz", function(e) {
  //     let ratio = e.loaded / e.total;
  //     redditProgressBarEl.value = ratio;
  //     redditProgressMbEl.innerHTML = Math.round(ratio*213)+" MB";
  //   });
  // }


  
  try {
    // existingEmbeddingsProgressCtn.style.display = "";
    
    embeddings = {};
    let file, opts;
    if(dataSource === "local") {
      file = await embeddingsFileHandle.getFile();
      opts = {};
    }
    if(dataSource === "online") {
      console.log("Trying to pick from online - but in the wrong place. Something's gone wrong...")
      // file = redditEmbeddingsBlob;
      // opts = {decompress:"gzip"};
      // add this back in for producton:
      file = await embeddingsFileHandle.getFile();
      opts = {};
    }
    
    let i = 0;
    for await (let line of makeTextFileLineIterator(file, opts)) {
      if(!line || !line.trim()) continue; // <-- to skip final new line (not sure if this is needed)
      // see how long line.split("\t") is 
      splitline = line.split("\t");
      if (splitline.length == 2) {
        let [filePath, embeddingVec] = splitline
      // console.log([filePath, embeddingVec])
      embeddings[filePath] = JSON.parse(embeddingVec);
      i++;
      }

    //   if(i % 1000 === 0) {
        // existingEmbeddingsLoadedEl.innerHTML = i;
        // await sleep(10);
    //   }
    }
  } catch(e) {
    // embeddings = undefined;
    console.log("No existing embedding found, or the embeddings file was corrupted:", e);
    // existingEmbeddingsProgressCtn.style.display = "none";
  }
  


    // hide #step1
    document.getElementById("step1").style.display = "none";
    // show #step2
    document.getElementById("step2").style.display = "block";
    // Set step2 class to hover 
    document.getElementById("step2").classList.add("hover");
    // Set timeout to remove hover class 
    setTimeout(function(){
      document.getElementById("step2").classList.remove("hover");
    }, 2000);




  // show searchSpinner
  document.getElementById("searchSpinner").style.display = "block";

  

    //  The end of pickdirectory - is always going to be....
    // Wait until window.vips is defined
    while (window.vips === undefined) {
      // console.log("waiting for vips")
      await sleep(100);
    }



    // Wait until embeddingsFileHandle is resolved

    await embeddingsFileHandle;

    if(dataSource === "local") {
      computeImageEmbeddings() // <-- this is the end of pickdirectory
    }

}


// STEP 2.5 - picking an existing embeddings file
async function getWebDataset(url) {


  // TODO - precalculate the TSNE in the gzipped file - would make things much quicker. 

  dataSource = "online"
  if (!workersInitialised){initializeWorkers();}


  const response = await fetch(url);
const jsonString = await response.text();

const onlineData = JSON.parse(jsonString);

console.log(onlineData)

// the data is jsonData[key].embeddings and jsonData[key] metadata - split into two dictionaries
// for each key in jsonData, add to embeddings and metadata dictionaries
embeddings = {}
metadata = {}
for (const [key, value] of Object.entries(onlineData)) {
    embeddings[key] = value
    // metadata[key] = value.metadata
}


//   webDataEmbeddingBlob = await downloadBlobWithProgress( url, function(e){console.log(e.loaded/e.total)})
//   file = webDataEmbeddingBlob;
// //   if url ends in .gz then decompress
// if(url.endsWith(".gz")) {
//   opts = {decompress:"gzip"};
// }
// else {
//     opts = {};
// }

//   embeddings = {};
//   metadata = {};
//   let i = 0;

//   for await (let line of makeTextFileLineIterator(file, opts)) {
//       if(!line || !line.trim()) continue; // <-- to skip final new line (not sure if this is needed)
//       splitline = line.split("\t");
//         let [filePath, embeddingVec, catalog] = splitline
//       embeddings[filePath] = JSON.parse(embeddingVec);
//       metadata[filePath] = JSON.parse(catalog);
//       i++;

      
//     }
//     // when finished, console log embeddings
//     // console.log(embeddings)
    console.log("Loaded precalculated embeddings ")


    if(document.getElementById("step1")){


    // hide #step1
    document.getElementById("step1").style.display = "none";
    // show #step2
    document.getElementById("step2").style.display = "block";
    // Set step2 class to hover 
    document.getElementById("step2").classList.add("hover");
    // Set timeout to remove hover class 
    setTimeout(function(){
      document.getElementById("step2").classList.remove("hover");
    }, 2000);

  }

    // hide localImagePanel
    // document.getElementById("localImagePanel").style.display = "none";
    

    // searchSort();

    // wait until onnxTextSession is defined
    while (onnxTextSession === undefined) {
      console.log("Loading model...")
      await sleep(500);
    }

    searchSort(); 


}



/////////////
//  STEP 3 //
/////////////
let totalEmbeddingsCount = 0;
let imagesEmbedded;
let recentEmbeddingTimes = []; // how long each embed took in ms, newest at end
let recomputeAllEmbeddings;
let imagesBeingProcessedNow = 0; 
let needToSaveEmbeddings = false;
async function computeImageEmbeddings() {

  // show computedEmbeddings

  console.log("Computing image embeddings...");
  imagesEmbedded = 0;
  totalEmbeddingsCount = Object.keys(embeddings).length;

  console.log("Got this number of embeddings: " + totalEmbeddingsCount)
  onlyEmbedNewImages = 1;

  recomputeAllEmbeddings = !onlyEmbedNewImages;
  let gotSomeExistingEmbeddings = totalEmbeddingsCount > 0;

  // Try: if not gotSomeExistingEmbeddings, then force recomputeAllEmbeddings to be true
  if (!gotSomeExistingEmbeddings) {
    console.log("forcing recompute")
    recomputeAllEmbeddings = true;
  }
  
  if(onlyEmbedNewImages && gotSomeExistingEmbeddings) {
    // preexistingEmbeddingsEl.display = "block";
    // preexistingEmbeddingsEl.innerHTML = `Loaded ${Object.keys(embeddings).length} preprocessed images.`; 
    // hide computedEmbeddings 
    // document.getElementById("computedEmbeddings").style.display = "none";
  }
  else {
    // preexistingEmbeddingsEl.display = "none";
    // document.getElementById("computedEmbeddings").style.display = "block";
  }

  if(recomputeAllEmbeddings || !gotSomeExistingEmbeddings) {
    embeddings = {}; // <-- maps file path (relative to top/selected directory) to embedding
  }

  // console.log(recomputeAllEmbeddings, gotSomeExistingEmbeddings, Object.keys(embeddings).length)
  
  try {
    await recursivelyProcessImagesInDir(directoryHandle);
    await saveEmbeddings();
  } catch(e) {
    console.error(e);
    alert(e.message);
  }

  // disableCtn(computeEmbeddingsCtnEl);
  // enableCtn(searchCtnEl);

  // hide loading spinner
  document.getElementById("searchSpinner").style.display = "none";

  console.log("Done computing image embeddings.");

  searchSort();

}


async function recursivelyProcessImagesInDir(dirHandle, currentPath="") {


  // console.log(dirHandle, currentPath)
      // image count first!!! 
            let imageCount = 0;

          // Count the number of image files
          for await (let [name, handle] of dirHandle) {
            const {kind} = handle;
            let path = `${currentPath}/${name}`;
           if(path.includes("/thumbnails")) continue;
            if (handle.kind === 'directory') {
              imageCount += await recursivelyProcessImagesInDir(handle, path);
            } else {
              // make lower case copy of path
              let pathLower = path.toLowerCase();

              let isImage = /\.(png|jpg|jpeg|webp|JPEG|JPG)$/.test(pathLower);
              if(!isImage) continue;

              imageCount++;
            }
          }

          // Print the total number of image files
          // console.log(`Total number of image files: ${imageCount}`);

          // If imageCount > maxNImages, then alert
          // if (imageCount > maxNImages) {
          //   alert(`You have selected a directory with ${imageCount} images. This is more than the maximum number of images recommended (1000).`);
          // }


          // set innerhtml of totalNumberImages to imageCount 
          // document.getElementById("totalNumberImages").innerHTML = imageCount;



  for await (let [name, handle] of dirHandle) {
    const {kind} = handle;
    let path = `${currentPath}/${name}`;
      // console.log(path)
      // ignore folder ./thumbnails/
      if(path.includes("/thumbnails")) continue;

    if ((handle.kind === 'directory') ) {
      await recursivelyProcessImagesInDir(handle, path);
    } else {
      // make lower case copy of path
      let pathLower = path.toLowerCase();


      let isImage = /\.(png|jpg|jpeg|webp|JPEG|JPG)$/.test(pathLower);
      if(!isImage) continue;

      // console.log("Processing image:", path)

      let alreadyGotEmbedding = !!embeddings[path];

      // console.log("Alreadygotembedding:",alreadyGotEmbedding)
      // console.log("Recompute:", recomputeAllEmbeddings)
      // console.log("needToSaveEmbeddings:", needToSaveEmbeddings)

      if(alreadyGotEmbedding && !recomputeAllEmbeddings) continue;
      
      if(needToSaveEmbeddings) {
        await saveEmbeddings();
        needToSaveEmbeddings = false;
      }
        
      while(imageWorkers.filter(w => !w.busy).length === 0) await sleep(1);
      
      let worker = imageWorkers.filter(w => !w.busy)[0];
      worker.busy = true;
      imagesBeingProcessedNow++;
      
      // if (Object.keys(embeddings).length >= maxNImages){continue}
      

      (async function() {

        if (Object.keys(embeddings).length >= imageCount){
          return;
        }

        // let startTime = Date.now();
        

        // try
        try{
        let blob = await handle.getFile();
        const embedVec = await modelData[MODEL_NAME].image.embed(blob, worker.session);
        // TODO - we can probably embed the path rather than the blob. This way we can use the embed function
        // to save the thumbnails, rather than computing all the the thumbs twice (as 224 for embedding and as 256 for the rendering). 
        // This is only an advantage when you have large images (otherwise you dont do the second step)

        // What we need to do:
        // define directoryHandleThumb  above the loop
        // define thumbnailPath within the loop
        // pass both directoryHandleThumb and thumbnailPath to image.embed function
        // image.embed then passes it to bicubicResizeAndCenterCrop() 
        // NB - that makes a centre crop (224x224), which is what we want for embedding but not for thumbnail display
        // So within the same canvas we might first draw a 256x256 image to save as a thumbnail, without cropping, then resize and crop and pass the blob back. 
        // Is this going to be any faster? It's not clear. We are at least limiting the number of read/write operations. 
        // However, we might do an alternative approach - leave the 4 workers to do the embedding, and start doing the thumbnailing in the meantime. 
        // By this point we do, after all, already have the paths to all images - see where imageCount is defined above. 

        embeddings[path] = [...embedVec];
        worker.busy = false;

        imagesEmbedded++;
        totalEmbeddingsCount++;
        }
        catch(e){
          console.log(e)
          console.log("Failed to process image ", path)
        worker.busy = false;
        }

        computeEmbeddingsProgressEl = document.getElementById("computeEmbeddingsProgress");
        computeEmbeddingsLoadingProgressBarEl = document.getElementById("computeEmbeddingsLoadingProgressBarEl");
        computeEmbeddingsText = document.getElementById("computeEmbeddingsText");
        

        // console.log(`Embedded ${Object.keys(embeddings).length} images in ${Date.now() - startTime} ms`);

        // computeEmbeddingsProgressEl.innerHTML = Object.keys(embeddings).length;

        // Update computeEmbeddingsLoadingProgressBarEl with the ratio of imagesEmbedded to imageCount
        if(imageCount){
        computeEmbeddingsLoadingProgressBarEl.value = Object.keys(embeddings).length / imageCount;
        // Make sure that the progress bar is visible
        computeEmbeddingsProgressEl.style.display = "block";
        computeEmbeddingsText.innerHTML = `Encoding images (${Object.keys(embeddings).length} of ${imageCount})`;
    
      }

        // If the ratio is not one, set step2 to :hover. Else remove :hover
        // maybe get rid of this
        // if ((Object.keys(embeddings).length < imageCount)&&(Object.keys(embeddings).length < maxNImages)){
        //   document.getElementById("step2").classList.add("hover");
        // } else {
        //   document.getElementById("step2").classList.remove("hover"); 
        // }

        

        
        let saveInterval = totalEmbeddingsCount > 50_000 ? 10_000 : 1000; // since saves take longer if there are lots of embeddings
        if(imagesEmbedded % saveInterval === 0) {
          needToSaveEmbeddings = true;
        }
        
        // recentEmbeddingTimes.push(Date.now()-startTime);
        // if(recentEmbeddingTimes.length > 100) recentEmbeddingTimes = recentEmbeddingTimes.slice(-50);
        // // if(recentEmbeddingTimes.length > 10) computeEmbeddingsSpeedEl.innerHTML = Math.round(recentEmbeddingTimes.slice(-20).reduce((a,v) => a+v, 0)/20);

        // // Compute the expected time left
        // let expectedTimeLeft = Math.round((imageCount - Object.keys(embeddings).length) * recentEmbeddingTimes.slice(-20).reduce((a,v) => a+v, 0)/20);
        // // convert to minutes and seconds
        // const expectedTimeString = `${Math.floor(expectedTimeLeft / 60000).toString().padStart(2, '0')}:${Math.floor((expectedTimeLeft % 60000) / 1000).toString().padStart(2, '0')}`;

        // if(recentEmbeddingTimes.length > 10) computeEmbeddingsSpeedEl.innerHTML = expectedTimeString;

        


        imagesBeingProcessedNow--;
      })();


    }
  }
  while(imagesBeingProcessedNow > 0) await sleep(10);
}


/////////////
//  STEP 4 //
/////////////

isRendered = false; 



  // Change this to only fire once - on loading the directory. Handlers can then change the x-y coordinates of the images based on new axis values. 
async function searchSort() {

  // if(searchBtn){searchBtn.disabled = true}
  
  // searchSpinner show
  // document.getElementById("searchSpinner").style.display = "block";
  
  
  // resultsEl.innerHTML = "Loading...";

  // while onnxTextSession not defined

  await sleep(2000);


  let resultHtml = "";
  let numResults = 0;
  // imageResults = [];



    // Then do UMAP stuff
    dataArray = [];
    orderedPath = [];
  for(let [path, embedding] of Object.entries(embeddings)) {
    // similarities[path] = cosineSimilarity(searchTextEmbedding, embedding);
    dataArray.push(embedding);
    orderedPath.push(path);
  }


  // searchTextEl.value = "A photo of Paris";
  // searchText = "A photo of Paris";
  // get from input box input_text 
  searchText = document.getElementById("input_text").value;

  // console.log(searchText);
  // console.log(modelData[MODEL_NAME].text)
          // hide div spinner-container 
          console.log('hiding spinner')
          document.getElementById("spinner-container").style.display = "none";

  // let searchTextEmbedding = await modelData[MODEL_NAME].text.embed(searchText, onnxTextSession);

  // console.log(searchTextEmbedding)
 
  const searchTextEmbeddingPromise = modelData[MODEL_NAME].text.embed(searchText, onnxTextSession);


  // Compute cosine similarities in parallel
  const embeddingsEntries = Object.entries(embeddings);
  const similaritiesPromises = embeddingsEntries.map(([path, embedding]) => {
    return Promise.all([searchTextEmbeddingPromise]).then(([searchTextEmbedding]) => {
      const similarity = cosineSimilarity(searchTextEmbedding, embedding);
      return [path, similarity]; 
    });
  });

  const similarityEntries = await Promise.all(similaritiesPromises);

  // console.log(similarityEntries )

  // for each similarity entry:
  // - get latitude longitude
  // - get score

  // remove all map polygons
  map.eachLayer(function (layer) {
    if (layer instanceof L.Polygon) {
        map.removeLayer(layer);
    }
  });


  minSimilarity = 1.0
  maxSimilarity = 0.0
  similarityEntries.forEach(([path, similarity]) => {
    if (similarity < minSimilarity) {minSimilarity = similarity}
    if (similarity > maxSimilarity) {maxSimilarity = similarity}
  });

  const similarityMap = similarityEntries.map(([path, similarity]) => {
    // path is of the form Paris/ParisImages/48.79108275862069_2.334958620689655_168.jpg
    // split on / and take the first element
    const latitude = parseFloat(path.split("/")[2].split("_")[0]);
    const longitude = parseFloat(path.split("/")[2].split("_")[1]);
    const score = similarity;
    // console.log(path)
    // console.log(similarity)
    // console.log(score)

  //   var circle = L.circle([latitude, longitude], {
  //     color: 'red',
  //     fillColor: getColor(score),
  //     fillOpacity: 0.5,
  //     radius: 200
  // }).addTo(map);

  

  geodelta = .00345;
  x1 = latitude - geodelta;
  x2 = latitude + geodelta;
  y1 = longitude - geodelta;
  y2 = longitude + geodelta;
  
  normScore = (score-minSimilarity)/(maxSimilarity-minSimilarity);
  // console.log(normScore)


  var mylayer = L.polygon([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], {
    // color: 'red',
    stroke:false,
    // fillColor: getColor(score),
    fillColor: interpolate("#e3e1bc",  "#c720c4", normScore),
    fillOpacity: 0.8,
    radius: 200
}).addTo(map);


// cbll= Latitude,longitude for Street View
// http://maps.google.com/maps?q=&layer=c&cbll=31.33519,-89.28720
// http://maps.google.com/maps?q=&layer=c&cbll=31.335198,-89.287204&cbp=
// cbp= Street View window that accepts 5 parameters:
// Street View/map arrangement, 11=upper half Street View and lower half map, 12=mostly Street View with corner map
// Rotation angle/bearing (in degrees)
// Tilt angle, -90 (straight up) to 90 (straight down)
// Zoom level, 0-2
// Pitch (in degrees) -90 (straight up) to 90 (straight down), default 5

mylayer.on('click', function() { 
  svurl = "http://maps.google.com/maps?q=&layer=c&cbll=" + latitude.toFixed(14) + ',' + longitude.toFixed(14)
  window.open(svurl, "_blank")

  })


  })


// var marker = L.marker([51.5, -0.09]).addTo(map);

   
}



/////////////////////////////
//  FUNCTIONS / UTILITIES  //
/////////////////////////////


function interpolate(color1, color2, percent) {
  // Convert the hex colors to RGB values
  const r1 = parseInt(color1.substring(1, 3), 16);
  const g1 = parseInt(color1.substring(3, 5), 16);
  const b1 = parseInt(color1.substring(5, 7), 16);

  const r2 = parseInt(color2.substring(1, 3), 16);
  const g2 = parseInt(color2.substring(3, 5), 16);
  const b2 = parseInt(color2.substring(5, 7), 16);

  // Interpolate the RGB values
  const r = Math.round(r1 + (r2 - r1) * percent);
  const g = Math.round(g1 + (g2 - g1) * percent);
  const b = Math.round(b1 + (b2 - b1) * percent);

  // Convert the interpolated RGB values back to a hex color
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function getColor(d) {
  return d > .23 ? '#800026' :
         d > .22  ? '#BD0026' :
         d > .21  ? '#E31A1C' :
         d > .20  ? '#FC4E2A' :
         d > .19   ? '#FD8D3C' :
         d > .18   ? '#FEB24C' :
         d > 0   ? '#FED976' :
                    '#FFEDA0';
}



function resetZoom(){
    controls.reset();
}


function normalizeGrid(imageResults){


  var imageData = [];

  // Ideal density is roughly 100 images for a 2x2 grid - thus 25 images / unit. 
  // We will scale the grid to fit the number of images we have.
  const idealDensity = 15;
  const density = imageResults.length / idealDensity;
  const scale = Math.sqrt(density);
  console.log("Scale: "+ scale + ", Density: " + density + ", Ideal Density: " + idealDensity)




  // Get the minimum and maximum values of score and score2
    const [minScore, maxScore] = d3.extent(imageResults, d => d.score);
    const [minScore2, maxScore2] = d3.extent(imageResults, d => d.score2);


    console.log("Range at start " + minScore + " " + maxScore + " " + minScore2 + " " + maxScore2)

    // Create a linear scale for score and score2
    const scaleScore = d3.scaleLinear()
      .domain([minScore, maxScore])
      .range([-scale, scale]);

    const scaleScore2 = d3.scaleLinear()
      .domain([minScore2, maxScore2])
      .range([-scale, scale]);



    for (let i = 0; i < imageResults.length; i++) {
    imageData.push({
      x: scaleScore(imageResults[i].score)  ,
      y: (scaleScore2(imageResults[i].score2) ),
      score: imageResults[i].score,
      score2: imageResults[i].score2,
      url: imageResults[i].url,
      path: imageResults[i].path,
      // also save the title as the end bit of the url
      // title: imageResults[i].url.split("/").slice(-1)[0],
      radius:20
    });
  }


    // Try force directed layout

    // centre the points so the median is at the centre
    const xMedian = d3.median(imageData, d => d.x);
    const yMedian = d3.median(imageData, d => d.y);
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].x -= xMedian;
      imageData[i].y -= yMedian;
    }

    // Save the old (x,y) pairs as orig_x and orig_y
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].orig_x = imageData[i].x;
      imageData[i].orig_y = imageData[i].y;
    }



    const collisionDistance = 0.12;

    // create a force-directed layout with repulsion, attraction, and collision forces
    const simulation = d3.forceSimulation(imageData)
      .force('collision', d3.forceCollide().radius(collisionDistance))
      .stop();
    // run the simulation for a set number of iterations
    nSimulation = 100;

    // if more than X images make Y steps
    if (imageData.length > 500) {
      nSimulation = 300;
    }

    for (let i = 0; i < nSimulation; i++) {
      simulation.tick(); 
    }

    // centre the post-simulated points so the median is at 0
    const medianX = d3.median(imageData, d => d.x);
    const medianY = d3.median(imageData, d => d.y);
    for (let i = 0; i < imageData.length; i++) {
      imageData[i].x -= medianX;
      imageData[i].y -= medianY;
    }


    // Save the new simulation (x,y) pairs as force_x, force_y

    for (let i = 0; i < imageData.length; i++) {
      imageData[i].force_x = imageData[i].x;
      imageData[i].force_y = imageData[i].y;
    }

    console.log("Range at end " + d3.extent(imageData, d => d.x) + " " + d3.extent(imageData, d => d.y))

    return imageData;
}

async function loadThumbnails(embeddings) {

  console.log("Loading thumbnails...")

  // If dataSource === "online" then the thumbnails are already loaded - just use the urls
  if (dataSource === "online") {
    const imageResults = Object.entries(embeddings).map(([path, embedding]) => {
      return {
        url: path,
        path: path,
        // catalog:catalog
        // metadata[path] contains the catalog - i.e. url to open on click
      };
    });
    return imageResults;
  }


// show progress bar
const progressBarContainer = document.getElementById("renderingProgressBar");
const progressBar = document.getElementById("renderingProgressBarEl");
const timeLeftSpan = document.getElementById("renderingTimeLeft");
progressBarContainer.style.display = "block";

const imageResults = [];

try {
directoryHandleThumb = await directoryHandle.getDirectoryHandle('thumbnails');
} catch {
directoryHandleThumb = await directoryHandle.getDirectoryHandle('thumbnails', { create: true });
}

// Check if thumbnailPath is in directoryHandleThumb
thumbFileNames = [];
for await (let [name,handle] of directoryHandleThumb){
thumbFileNames.push(name);
}

// create an array of promises for loading all thumbnails
const promises = Object.entries(embeddings).map(async ([path, embedding]) => {
let url;
let handle;

// // index 
// const i = Object.keys(embeddings).indexOf(path);
// const score = umap_embedding[i][0];
// const score2 = umap_embedding[i][1];

// if path is not already a URL
if (!path.startsWith("http")) {
// console.log("Getting thumb " , path)
url = await getThumbnail(path, directoryHandleThumb, thumbFileNames);
} else {
url = path;
}

imageResults.push({ url, path });

// update progress bar
progressBar.value += 1;

// // calculate estimated time remaining
// const progress = progressBar.value / progressBar.max;
// const elapsedTime = (Date.now() - startTime) / 1000;
// const estimatedTimeRemaining = (elapsedTime / progress) - elapsedTime;

// // update progress bar label
// const timeLeft = formatTime(estimatedTimeRemaining);
// timeLeftSpan.innerHTML = `Time left: ${timeLeft}`;

});

// initialize progress bar
progressBar.max = promises.length;
progressBar.value = 0;

// record start time
// const startTime = Date.now();

// wait for all promises to resolve
await Promise.all(promises);

// hide progress bar
progressBarContainer.style.display = "none";

return imageResults;
}

function formatTime(seconds) {
const minutes = Math.floor(seconds / 60);
const remainingSeconds = Math.floor(seconds % 60);
return `${minutes < 10 ? '0' : ''}${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}


function clearD3(){
  d3.select("svg").remove();
}



async function getThumbnail(path, directoryHandleThumb, thumbFileNames, maxSize = 0.2) {
// Roughly 10 kB - thus 0.01 megabytes - for a 256x256 image. Only bother if it is more than 20 times bigger, i.e. more than .2 MB
  const fileName = path.split("/").pop();
const thumbnailPath = `thumb_${fileName}`;  

const thumbnailExists = thumbFileNames.includes(thumbnailPath);
// console.log("thumbnailExists", thumbnailPath);


if ( thumbnailExists) {
// load thumbnail from file system
const thumbnailFile = await getThumbnailFile(directoryHandleThumb,thumbnailPath);
return URL.createObjectURL(thumbnailFile);
}


// check if thumbnail already exists in .thumbnails folder
// const thumbnailPath = `./.thumbnails/${handle.name}`;


const handle = await getFileHandleByPath(path);
const maxDimension = 256;
const blob = await handle.getFile();
const sizeInMB = blob.size / (1024 * 1024);

if (sizeInMB <= maxSize){
// console.log('loaded from file system')
// load image direct from file system
return URL.createObjectURL(blob);
}

// console.log('creating thumbnail')

const image = new Image();
image.src = URL.createObjectURL(blob);
const canvas = document.createElement('canvas');
const context = canvas.getContext('2d');



// change directory handle to 'thumbnails' subfolder

//  console.log("Created/found thumbnails folder")

return new Promise((resolve) => {
image.onload = async () => {
const { width, height } = image;
canvas.width = width;
canvas.height = height;
context.drawImage(image, 0, 0);

let thumbnailWidth = width;
let thumbnailHeight = height;

if (thumbnailWidth > maxDimension) {
  thumbnailWidth = maxDimension;
  thumbnailHeight = Math.round(height * maxDimension / width);
}

if (thumbnailHeight > maxDimension) {
  thumbnailHeight = maxDimension;
  thumbnailWidth = Math.round(width * maxDimension / height);
}

const thumbnailCanvas = document.createElement('canvas');
const thumbnailContext = thumbnailCanvas.getContext('2d');
thumbnailCanvas.width = thumbnailWidth;
thumbnailCanvas.height = thumbnailHeight;
thumbnailContext.drawImage(canvas, 0, 0, width, height, 0, 0, thumbnailWidth, thumbnailHeight);

const thumbnailDataURL = thumbnailCanvas.toDataURL('image/jpeg', 0.8);

// save thumbnail to file system
await saveThumbnailFile(directoryHandleThumb, thumbnailPath, thumbnailCanvas);

resolve(thumbnailDataURL);
};
});
}

async function fileExists(path) {
try {
const file = await  stat(path);
return file.isFile();
} catch {
return false;
}
}

async function getThumbnailFile(dirHand, path) {
const handle = await  dirHand.getFileHandle(path, { create: false });
return await handle.getFile();
}

async function saveThumbnailFile(dirHand, path, canvas) {
// create file and write data to it
const fileHandle = await dirHand.getFileHandle(path, { create: true });
const writable = await fileHandle.createWritable();
canvas.toBlob(async function (blob) {
await writable.write(blob);
await writable.close();
}, 'image/jpeg', 0.8);
}


function sanitizeFilename(filename) {
return filename.replace(/[^a-zA-Z0-9-_.]/g, "_");
}



async function getFileHandleByPath(path) {
  let handle = directoryHandle;
  let chunks = path.split("/").slice(1);
  for(let i = 0; i < chunks.length; i++) {
    let chunk = chunks[i];
    if(i === chunks.length-1) {
      handle = await handle.getFileHandle(chunk);
    } else {
      handle = await handle.getDirectoryHandle(chunk);
    }
  }
  return handle;
}

async function getRgbData(blob) { 
  // let blob = await fetch(imgUrl, {referrer:""}).then(r => r.blob());

  let resizedBlob = await bicubicResizeAndCenterCrop(blob);
  let img = await createImageBitmap(resizedBlob);

  let oscanvas = new OffscreenCanvas(224, 224);
  let ctx = oscanvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  let imageData = ctx.getImageData(0, 0, oscanvas.width, oscanvas.height);

  let rgbData = [[], [], []]; // [r, g, b]
  // remove alpha and put into correct shape:
  let d = imageData.data;
  for(let i = 0; i < d.length; i += 4) { 
    let x = (i/4) % oscanvas.width;
    let y = Math.floor((i/4) / oscanvas.width)
    if(!rgbData[0][y]) rgbData[0][y] = [];
    if(!rgbData[1][y]) rgbData[1][y] = [];
    if(!rgbData[2][y]) rgbData[2][y] = [];
    rgbData[0][y][x] = d[i+0]/255;
    rgbData[1][y][x] = d[i+1]/255;
    rgbData[2][y][x] = d[i+2]/255;
    // From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954;
    rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258;
    rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711;
  }
  rgbData = Float32Array.from(rgbData.flat().flat());
  return rgbData;
}

async function bicubicResizeAndCenterCrop(blob) {
  let im1 = vips.Image.newFromBuffer(await blob.arrayBuffer());

  // Resize so smallest side is 224px:
  const scale = 224 / Math.min(im1.height, im1.width);
  let im2 = im1.resize(scale, { kernel: vips.Kernel.cubic });

  // crop to 224x224:
  let left = (im2.width - 224) / 2;
  let top = (im2.height - 224) / 2;
  let im3 = im2.crop(left, top, 224, 224)

  let outBuffer = new Uint8Array(im3.writeToBuffer('.png'));
  im1.delete(), im2.delete(), im3.delete();
  return new Blob([outBuffer], { type: 'image/png' });
}


function downloadBlobWithProgressOld(url, onProgress) {
  return new Promise((res, rej) => {
    var blob;
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'arraybuffer';
    xhr.onload = function(e) {
      blob = new Blob([this.response]);   
    };
    xhr.onprogress = onProgress;
    xhr.onloadend = function(e){
      res(blob);
    }
    xhr.send();
  });
}


function downloadBlobWithProgress(url, onProgress) {
return new Promise((res, rej) => {
const filename = url.substring(url.lastIndexOf('/')+1);
const dbRequest = window.indexedDB.open('myDatabase', 1);
dbRequest.onerror = rej;
dbRequest.onupgradeneeded = function(event) {
const db = event.target.result;
db.createObjectStore('files');
};
dbRequest.onsuccess = function(event) {
const db = event.target.result;
const tx = db.transaction(['files'], 'readonly');
const store = tx.objectStore('files');
const getRequest = store.get(filename);
getRequest.onsuccess = function(event) {
  const fileData = event.target.result;
  if (fileData) {
    // file already exists in IndexedDB, load from dataURL
    const blob = dataURLToBlob(fileData);
    res(blob);
  } else {
    // file does not exist in IndexedDB, download and save
    const xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'blob';
    xhr.onload = function(e) {
      const blob = this.response;
      const reader = new FileReader();
      reader.onloadend = function() {
        const tx = db.transaction(['files'], 'readwrite');
        const store = tx.objectStore('files');
        store.put(reader.result, filename);
      };
      reader.readAsDataURL(blob);
      res(blob);
    };
    xhr.onprogress = onProgress;
    xhr.onerror = rej;
    xhr.send();
  }
};
};
});
}

function dataURLToBlob(dataURL) {
const arr = dataURL.split(',');
const mime = arr[0].match(/:(.*?);/)[1];
const bstr = atob(arr[1]);
let n = bstr.length;
const u8arr = new Uint8Array(n);
while(n--) {
u8arr[n] = bstr.charCodeAt(n);
}
return new Blob([u8arr], {type:mime});
}
// end of IndexedDB code

async function saveEmbeddings(opts={}) {
  let writable = await embeddingsFileHandle.createWritable();
  let textBatch = "";
  let i = 0;
  for(let [filePath, embeddingVec] of Object.entries(embeddings)) {
    let vecString = opts.compress ? JSON.stringify(embeddingVec.map(n => n.toFixed(3))).replace(/"/g, "") : JSON.stringify(embeddingVec);
    textBatch += `${filePath}\t${vecString}\n`;
    i++;
    if(i % 1000 === 0) {
      await writable.write(textBatch);
      textBatch = "";
    }
  }
  await writable.write(textBatch);
  await writable.close();
}

// Tweaked version of example from here: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStreamDefaultReader/read
async function* makeTextFileLineIterator(blob, opts={}) {
  const utf8Decoder = new TextDecoder("utf-8");
  let stream = await blob.stream();
  
  if(opts.decompress === "gzip") stream = stream.pipeThrough(new DecompressionStream("gzip"));
  
  let reader = stream.getReader();
  
  let {value: chunk, done: readerDone} = await reader.read();
  chunk = chunk ? utf8Decoder.decode(chunk, {stream: true}) : "";

  let re = /\r\n|\n|\r/gm;
  let startIndex = 0;

  while (true) {
    let result = re.exec(chunk);
    if (!result) {
      if (readerDone) {
        break;
      }
      let remainder = chunk.substr(startIndex);
      ({value: chunk, done: readerDone} = await reader.read());
      chunk = remainder + (chunk ? utf8Decoder.decode(chunk, {stream: true}) : "");
      startIndex = re.lastIndex = 0;
      continue;
    }
    yield chunk.substring(startIndex, result.index);
    startIndex = re.lastIndex;
  }
  if (startIndex < chunk.length) {
    // last line didn't end in a newline char
    yield chunk.substr(startIndex);
  }
}

function cosineSimilarity(A, B) {
  if(A.length !== B.length) throw new Error("A.length !== B.length");
  let dotProduct = 0, mA = 0, mB = 0;
  for(let i = 0; i < A.length; i++){
    dotProduct += A[i] * B[i];
    mA += A[i] * A[i];
    mB += B[i] * B[i];
  }
  mA = Math.sqrt(mA);
  mB = Math.sqrt(mB);
  let similarity = dotProduct / (mA * mB);
  return similarity;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function enableCtn(el) {
  el.style.opacity = 1;
  el.style.pointerEvents = "";
}
function disableCtn(el) {
  el.style.opacity = 0.5;
  el.style.pointerEvents = "none";
}


function getCSV() {
// We want 3 columns to our CSV: filename, prompt1 (score), and prompt 2 (score2).
// searchtextel
const prompt1 = document.getElementById('searchTextEl').value;
const prompt2 = document.getElementById('searchTextEl2').value;

// If prompt1 or prompt2 are empty, replace with x-axis or y-axis
prompt1Label = prompt1 || 'x-axis-UMAP';
prompt2Label = prompt2 || 'y-axis-UMAP';

let csv = 'Image,' + prompt1Label + ',' + prompt2Label + '\n';

// Iterate through the d3 data and add a row for each file, with d.score and d.score2 
// as the prompt1 and prompt2 scores.
for (let i = 0; i < imageResults.length; i++) {
const d = imageResults[i];
// Strip any commas from d.path 
const path = d.path.replace(/,/g, '');
csv += path + ',' + d.score + ',' + d.score2 + '\n';
}

// Create a Blob object from the CSV data.
const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });

// Create a URL for the Blob object using the createObjectURL() method.
const url = URL.createObjectURL(blob);

// Create a link element and set its attributes.
const link = document.createElement('a');
link.setAttribute('href', url);
link.setAttribute('download', '2DCLIP.csv');

// Trigger a click event on the link element to download the CSV file.
link.click();
}



{/* </script> */}