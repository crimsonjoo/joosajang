$(document).mousemove(function (event) {
  cx = Math.ceil($(window).width() / 1.8);
  cy = Math.ceil($(window).height() / 1.8);
  dx = event.pageX - cx;
  dy = event.pageY - cy;

  tiltx = dy / cy;
  tilty = -(dx / cx);
  radius = Math.sqrt(Math.pow(tiltx, 2) + Math.pow(tilty, 2));
  degree = radius * 20;

  $("#head").css(
    "-webkit-transform",
    "rotate3d(" + -tiltx + ", " + -tilty + ", 0, " + degree + "deg)"
  );
  $("#head").css(
    "transform",
    "rotate3d(" + -tiltx + ", " + -tilty + ", 0, " + degree + "deg)"
  );
});
function changeAnimState() {
  $(".eye-left").clone(true).insertBefore(".eye-left");
  $(".eye-left").last().remove();
  $(".eye-right").clone(true).insertBefore(".eye-right");
  $(".eye-right").last().remove();
}

setInterval(changeAnimState, 3000);
