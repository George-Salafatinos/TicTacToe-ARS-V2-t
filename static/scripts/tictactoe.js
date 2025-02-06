console.log("tictactoe.js loaded");

let currentBoard = ["", "", "", "", "", "", "", "", ""];
let currentPlayer = "X";
let selectedAlgorithm = "";
let gameOver = false;

async function trainModel() {
  const algorithmSelect = document.getElementById("algorithm-select");
  const opponentSelect = document.getElementById("opponent-select");
  const modelNameInput = document.getElementById("model-name-input");
  const stepsInput = document.getElementById("steps-input");
  const lrInput = document.getElementById("lr-input");
  const gammaInput = document.getElementById("gamma-input");

  const trainStatus = document.getElementById("train-status");
  const trainingChart = document.getElementById("training-chart");

  selectedAlgorithm = algorithmSelect.value;
  const opponent = opponentSelect.value;
  const modelName = modelNameInput.value.trim() || "unnamed";
  const steps = parseInt(stepsInput.value) || 10;
  const lr = lrInput.value.trim() || "0.01";
  const gamma = gammaInput.value.trim() || "0.99";

  if (!selectedAlgorithm) {
    trainStatus.textContent = "Please select an algorithm before training.";
    return;
  }

  trainStatus.textContent = "Training in progress...";
  trainingChart.style.display = "none";

  try {
    const response = await fetch("/train-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: selectedAlgorithm,
        hyperparams: {
          steps: steps,
          model_name: modelName,
          opponent: opponent,
          lr: lr,
          gamma: gamma
        }
      })
    });
    const data = await response.json();
    trainStatus.textContent = data.message || "Training complete.";
    if (data.chart_b64) {
      trainingChart.src = "data:image/png;base64," + data.chart_b64;
      trainingChart.style.display = "block";
    }
    console.log("Training details:", data.details);
  } catch (error) {
    console.error("Error during training:", error);
    trainStatus.textContent = "Error during training. See console.";
  }
}

async function selectModel() {
  const playStatus = document.getElementById("play-status");
  if (!selectedAlgorithm) {
    playStatus.textContent = "No model trained or selected. Train first.";
    return;
  }
  try {
    const resp = await fetch("/select-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm: selectedAlgorithm })
    });
    const data = await resp.json();
    if (data.algorithm) {
      selectedAlgorithm = data.algorithm;
    }
    playStatus.textContent = data.message || "Model selected.";
  } catch (err) {
    console.error("Error selecting model:", err);
    playStatus.textContent = "Error selecting model. See console.";
  }
}

async function loadModelList() {
  const selectEl = document.getElementById("model-file-select");
  selectEl.innerHTML = '<option value="" disabled selected>-- fetching models --</option>';
  try {
    const resp = await fetch("/list-models");
    const data = await resp.json();
    if (!Array.isArray(data.files)) {
      selectEl.innerHTML = '<option value="" disabled selected>-- no files --</option>';
      return;
    }
    selectEl.innerHTML = '';
    data.files.forEach(file => {
      const opt = document.createElement("option");
      opt.value = file;
      opt.textContent = file;
      selectEl.appendChild(opt);
    });
  } catch (error) {
    console.error("Error loading model list:", error);
    selectEl.innerHTML = '<option value="" disabled selected>-- error --</option>';
  }
}

async function loadDiskModel() {
  const playStatus = document.getElementById("play-status");
  const selectEl = document.getElementById("model-file-select");
  const filename = selectEl.value;
  if (!filename) {
    playStatus.textContent = "Please select a model file first.";
    return;
  }
  try {
    const resp = await fetch("/select-model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: filename })
    });
    const data = await resp.json();
    if (data.algorithm) {
      selectedAlgorithm = data.algorithm;
    }
    playStatus.textContent = data.message || "Model loaded from disk.";
  } catch (error) {
    console.error("Error loading disk model:", error);
    playStatus.textContent = "Error loading disk model. See console.";
  }
}

function onCellClick(event) {
  const cell = event.target;
  const index = parseInt(cell.dataset.index);
  if (gameOver || currentBoard[index] !== "" || currentPlayer !== "X") {
    return;
  }
  currentBoard[index] = "X";
  cell.textContent = "X";
  if (!checkGameStatus("X")) {
    currentPlayer = "O";
    modelMove();
  }
}

async function modelMove() {
  const playStatus = document.getElementById("play-status");
  try {
    const response = await fetch("/model-move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm: selectedAlgorithm,
        board: currentBoard
      })
    });
    const data = await response.json();
    if (data.error) {
      playStatus.textContent = "Model move error: " + data.error;
      currentPlayer = "X";
      return;
    }
    const moveIndex = data.move;
    if (moveIndex !== null && moveIndex !== undefined) {
      currentBoard[moveIndex] = "O";
      const cell = document.getElementById(`cell-${moveIndex}`);
      if (cell) {
        cell.textContent = "O";
      }
    }
    checkGameStatus("O");
    if (!gameOver) {
      currentPlayer = "X";
    }
  } catch (error) {
    console.error("Error during model move:", error);
    playStatus.textContent = "Error during model move. See console.";
    currentPlayer = "X";
  }
}

function checkGameStatus(playerMoved) {
  const playStatus = document.getElementById("play-status");
  const winner = checkWinner(currentBoard);
  if (winner) {
    if (winner === "X") {
      playStatus.textContent = "You Win!";
    } else {
      playStatus.textContent = "You Lose!";
    }
    gameOver = true;
    return true;
  }
  if (!currentBoard.includes("")) {
    playStatus.textContent = "It's a draw.";
    gameOver = true;
    return true;
  }
  return false;
}

function checkWinner(board) {
  const lines = [
    [0,1,2], [3,4,5], [6,7,8],
    [0,3,6], [1,4,7], [2,5,8],
    [0,4,8], [2,4,6]
  ];
  for (const [a,b,c] of lines) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) {
      return board[a];
    }
  }
  return null;
}

function resetBoard() {
  currentBoard = ["", "", "", "", "", "", "", "", ""];
  currentPlayer = "X";
  gameOver = false;
  const playStatus = document.getElementById("play-status");
  playStatus.textContent = "Board reset. Make a move!";
  document.querySelectorAll(".cell").forEach(cell => {
    cell.textContent = "";
  });
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("train-button").addEventListener("click", trainModel);
  document.getElementById("select-model-button").addEventListener("click", selectModel);
  document.getElementById("load-model-list-button").addEventListener("click", loadModelList);
  document.getElementById("load-disk-model-button").addEventListener("click", loadDiskModel);
  document.getElementById("reset-button").addEventListener("click", resetBoard);

  document.querySelectorAll(".cell").forEach(cell => {
    cell.addEventListener("click", onCellClick);
  });
});
