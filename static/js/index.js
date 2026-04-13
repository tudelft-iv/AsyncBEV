window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function setupSynchronizedVideos() {
  var videos = Array.from(document.querySelectorAll('video[data-sync-group]'));
  if (videos.length < 2) {
    return;
  }

  var syncing = false;

  function inSameGroup(source, target) {
    return source.dataset.syncGroup && source.dataset.syncGroup === target.dataset.syncGroup;
  }

  function syncOthers(source, action) {
    if (syncing) {
      return;
    }
    syncing = true;
    videos.forEach(function(target) {
      if (target === source || !inSameGroup(source, target)) {
        return;
      }

      action(target);
    });
    syncing = false;
  }

  videos.forEach(function(video) {
    video.addEventListener('play', function() {
      var sourceTime = video.currentTime;
      var sourceRate = video.playbackRate;
      syncOthers(video, function(target) {
        target.currentTime = sourceTime;
        target.playbackRate = sourceRate;
        target.play();
      });
    });

    video.addEventListener('pause', function() {
      syncOthers(video, function(target) {
        target.pause();
      });
    });

    video.addEventListener('seeking', function() {
      var sourceTime = video.currentTime;
      syncOthers(video, function(target) {
        target.currentTime = sourceTime;
      });
    });

    video.addEventListener('ratechange', function() {
      var sourceRate = video.playbackRate;
      syncOthers(video, function(target) {
        target.playbackRate = sourceRate;
      });
    });
  });
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    setupSynchronizedVideos();

    bulmaSlider.attach();

})
