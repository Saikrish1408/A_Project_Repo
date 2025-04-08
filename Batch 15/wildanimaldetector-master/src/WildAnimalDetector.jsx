import React, { useRef, useEffect, useState } from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "@tensorflow/tfjs";

const TWILIO_SID = "AC05f8709abe4c636868dac9a9fb907bd6";
const TWILIO_AUTH = "46119bc47df8f6af4f2b52d3ab97c8df";
const TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"; // Twilio Sandbox Number
const YOUR_PHONE_NUMBER = "whatsapp:+919943446468";

const WildAnimalDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [animalName, setAnimalName] = useState("Detecting...");
  
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing the camera:", error);
      }
    };

    startCamera();
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      const model = await mobilenet.load();

      setInterval(async () => {
        if (videoRef.current) {
          const predictions = await model.classify(videoRef.current);

          if (predictions.length > 0) {
            const topPrediction = predictions[0].className;
            setAnimalName(isWildAnimal(topPrediction) ? topPrediction : "No wild animal detected");

            if (isWildAnimal(topPrediction)) {
              const imageData = captureImage();
              saveToTextFile(topPrediction, imageData);
              sendWhatsAppMessage(topPrediction);
            }
          }
        }
      }, 5000);
    };

    loadModel();
  }, []);

  const isWildAnimal = (label) => {
    const wildAnimals = ["lion", "tiger", "elephant", "bear", "wolf", "wild pig", "boar","dog","cat","cow"];
    return wildAnimals.some((animal) => label.toLowerCase().includes(animal));
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return null;

    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/png");
  };

  const saveToTextFile = (animal, imageData) => {
    const timestamp = new Date().toLocaleString();
    const data = `Time: ${timestamp}\nAnimal: ${animal}\nImage: ${imageData}\n\n`;
    
    const blob = new Blob([data], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "wild_animal_detection.txt";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const sendWhatsAppMessage = (animal) => {
    fetch("https://api.twilio.com/2010-04-01/Accounts/" + TWILIO_SID + "/Messages.json", {
      method: "POST",
      headers: {
        "Authorization": "Basic " + btoa(TWILIO_SID + ":" + TWILIO_AUTH),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        From: TWILIO_WHATSAPP_NUMBER,
        To: YOUR_PHONE_NUMBER,
        Body: `🚨 Wild Animal Alert!\nAnimal: ${animal}\nTime: ${new Date().toLocaleString()}`
      }),
    })
      .then((res) => res.json())
      .then((data) => console.log("WhatsApp Message Sent:", data))
      .catch((err) => console.error("WhatsApp Error:", err));
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h2>Wild Animal Detector</h2>
      <video ref={videoRef} autoPlay playsInline style={{ width: "100%", maxWidth: "600px", border: "2px solid black" }}></video>
      <h3>{animalName}</h3>

      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
    </div>
  );
};

export default WildAnimalDetector;
