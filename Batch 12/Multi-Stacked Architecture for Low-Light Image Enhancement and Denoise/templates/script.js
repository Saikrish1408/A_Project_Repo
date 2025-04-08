
document.addEventListener('contextmenu', (e) => e.preventDefault());

function ctrlShiftKey(e, keyCode) {
  return e.ctrlKey && e.shiftKey && e.keyCode === keyCode.charCodeAt(0);
}

document.onkeydown = (e) => {
  // Disable F12, Ctrl + Shift + I, Ctrl + Shift + J, Ctrl + U
  if (
    event.keyCode === 123 ||
    ctrlShiftKey(e, 'I') ||
    ctrlShiftKey(e, 'J') ||
    ctrlShiftKey(e, 'C') ||
    (e.ctrlKey && e.keyCode === 'U'.charCodeAt(0))
  )
    return false;
};


function descriptionHeader(){
  let dhi = 0;
  let dhtxt = 'Looking for a great encryption mechanism?';
  let dhtxtArr =dhtxt.split(""); 
  let dhspeed = 50;
  let dhtemp=" ";
  function descriptionHeaderWrite(){
    if (dhi < dhtxt.length) {
      document.getElementById("descriptionHeader").innerText = dhtemp + dhtxtArr[dhi]+ "|";
      dhtemp = dhtemp + dhtxtArr[dhi];
      dhi++;
      setTimeout(descriptionHeaderWrite, dhspeed);
    }else{
      document.getElementById("descriptionHeader").innerHTML = 'Looking for a great <mark>encryption mechanism?</mark>';
      descriptionSubHeader()
    }
  }
  descriptionHeaderWrite();
}

function descriptionSubHeader(){
  let dshi = 0;
  let dshtxt = `Then don't worry buddy you are in a correct place. Upload your confidentails text file in our portal and get it encrypted to the top most level. Our encryption mechanism is powered by Shor and QPE mechanism.`;
  let dshtxtArr =dshtxt.split(""); 
  let dshspeed = 40;
  let dshtemp=" ";
  function descriptionSubHeaderWrite(){
    if (dshi < dshtxt.length) {
      document.getElementById("descriptionSubHeader").innerText = dshtemp + dshtxtArr[dshi]+ "|";
      dshtemp = dshtemp + dshtxtArr[dshi];
      dshi++;
      setTimeout(descriptionSubHeaderWrite, dshspeed);
    }else{
      document.getElementById("encrypt_btn").style.display="inline";
      projectSectionHeader();
    }
  }
  descriptionSubHeaderWrite()
}


descriptionHeader()