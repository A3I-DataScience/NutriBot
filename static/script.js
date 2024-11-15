let lightMode = true;
let recorder = null;
let recording = false;
let profileOption = "unknown";
let genderOption = "unknown";
let ageOption = "unknown";
let sizeOption = "unknown";
let weightOption = "unknown";
let countryOption = "unknown";
let optionChanges = {};

const responses = [];
const botRepeatButtonIDToIndexMap = {};
const userRepeatButtonIDToRecordingMap = {};
const baseUrl = window.location.origin


const areOptionsValid = () => {
  // Condition 1: `profileOption` is filled
  if (profileOption && profileOption !== "unknown") {
    return true;
  }

  // Condition 2: All other options are filled
  const otherOptions = [genderOption, ageOption, sizeOption, weightOption, countryOption];
  return otherOptions.every(option => option && option !== "unknown");
};
const updateSendButtonState = () => {
  const sendButton = document.getElementById("send-button");
  const messageInput = document.getElementById("message-input");
  
  // Enable or disable the send button based on areOptionsValid()
  sendButton.disabled = !areOptionsValid();
  
};

$("#send-button").click(async function () {
  if (!areOptionsValid()) {
    alert("Please complete your profile or fill out all the other fields.");
    return;
  }

  const message = cleanTextInput($("#message-input").val());
  populateUserMessage(message, null);
  populateBotResponse(message);
});


async function showBotLoadingAnimation() {
  await sleep(200);
  $(".loading-animation")[1].style.display = "inline-block";
  document.getElementById('send-button').disabled = true;
}

function hideBotLoadingAnimation() {
  $(".loading-animation")[1].style.display = "none";
  if(!isFirstMessage){
    document.getElementById('send-button').disabled = false;
  }
}

async function showUserLoadingAnimation() {
  await sleep(100);
  $(".loading-animation")[0].style.display = "flex";
}

function hideUserLoadingAnimation() {
  $(".loading-animation")[0].style.display = "none";
}


const processUserMessage = async (userMessage) => {
  let response = await fetch(baseUrl + "/process-message", {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify({ userMessage: userMessage,
      profile: profileOption,
      gender: genderOption,
      age: ageOption,
      size: sizeOption,
      weight: weightOption,
      country: countryOption,
     }),
  });
  response = await response.json();
  console.log(response);
  return response;
};

const cleanTextInput = (value) => {
  return value
    .trim() // remove starting and ending spaces
    .replace(/[\n\t]/g, "") // remove newlines and tabs
    .replace(/<[^>]*>/g, "") // remove HTML tags
    .replace(/[<>&;]/g, ""); // sanitize inputs
};

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));

const scrollToBottom = () => {
  // Scroll the chat window to the bottom
  $("#chat-window").animate({
    scrollTop: $("#chat-window")[0].scrollHeight,
  });
};

const populateUserMessage = (userMessage, userRecording) => {
  // Clear the input field
  $("#message-input").val("");

  // Append the user's message to the message list
    $("#message-list").append(
      `<div class='message-line my-text'><div class='message-box my-text${
        !lightMode ? " dark" : ""
      }'><div class='me'>${userMessage}</div></div></div>`
    );

  scrollToBottom();
};

let isFirstMessage = true;

const populateBotResponse = async (userMessage) => {
  await showBotLoadingAnimation();

  let response;

  if (isFirstMessage) {
    response = { botResponse: "Hello there! I'm NutriBot your nutrition assistant. I will ask you a few questions to understand you better and provide you with personalized nutrition advice. Let's get started! ok?" };
    
    renderBotResponse(response);
    isFirstMessage = false;
    return;
  }

  // Generate a message for the changed options, if any
  if (Object.keys(optionChanges).length > 0) {
    let changesMessage = "";
    for (const [key, value] of Object.entries(optionChanges)) {
      changesMessage = `I see you've updated your profile, please give me some time to think. `;
    }

    // Render acknowledgment for changes
    renderBotResponse({ botResponse: changesMessage });

    optionChanges = {}; // Reset changes after acknowledgment
  }

  await showBotLoadingAnimation();

  // Process the user's message and render the bot's response
  response = await processUserMessage(userMessage);
  renderBotResponse(response);
};

const renderBotResponse = (response) => {
  responses.push(response);

  hideBotLoadingAnimation();

  $("#message-list").append(
    `<div class='message-line'><div class='message-box${!lightMode ? " dark" : ""}'>${response.botResponse.trim()}<br></div></div>`
  );

  scrollToBottom();
}

populateBotResponse()


$(document).ready(function () {

  //start the chat with send button disabled
  document.getElementById('send-button').disabled = true;

  // Listen for the "Enter" key being pressed in the input field
  $("#message-input").keyup(function (event) {
    let inputVal = cleanTextInput($("#message-input").val());

    if (event.keyCode === 13 && inputVal != "") {
      const message = inputVal;

      populateUserMessage(message, null);
      populateBotResponse(message);
    }

    inputVal = $("#message-input").val();
  });

  // When the user clicks the "Send" button
  $("#send-button").click(async function () {
  // Get the message the user typed in
  const message = cleanTextInput($("#message-input").val());

  populateUserMessage(message, null);
  populateBotResponse(message);

  });

  //reset chat
  // When the user clicks the "Reset" button
    $("#reset-button").click(async function () {
      // Clear the message list
      $("#message-list").empty();

      // Reset the responses array
      responses.length = 0;

      // Reset isFirstMessage flag
      isFirstMessage = true;

      document.querySelector('#upload-button').disabled = false;

      // Start over
      populateBotResponse();
    });


  // handle the event of switching light-dark mode
  $("#darkmode-toggle").change(function () {
    $("body").toggleClass("dark-mode");
    $(".message-box").toggleClass("dark");
    $(".loading-dots").toggleClass("dark");
    $(".dot").toggleClass("dark-dot");
    lightMode = !lightMode;
  });

 $("#profile-options").change(function () {
    profileOption = $(this).val();
    console.log("Profile option changed to:", profileOption);
    optionChanges["Profile"] = profileOption;
    updateSendButtonState();
});

$("#gender-options").change(function () {
    genderOption = $(this).val();
    console.log("Gender option changed to:", genderOption);
    optionChanges["Gender"] = genderOption;
    updateSendButtonState();
});

$("#age-options").change(function () {
    ageOption = $(this).val();
    console.log("Age option changed to:", ageOption);
    optionChanges["Age"] = ageOption;
    updateSendButtonState();
});

$("#size-options").change(function () {
    sizeOption = $(this).val();
    console.log("Size option changed to:", sizeOption);
    optionChanges["Size"] = sizeOption;
    updateSendButtonState();
});

$("#weight-options").change(function () {
    weightOption = $(this).val();
    console.log("Weight option changed to:", weightOption);
    optionChanges["Weight"] = weightOption;
    updateSendButtonState();
});

$("#country-options").change(function () {
    countryOption = $(this).val();
    console.log("Country option changed to:", countryOption);
    optionChanges["Country"] = countryOption;
    updateSendButtonState();
});


});
