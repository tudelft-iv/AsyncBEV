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

  var PROGRAMMATIC_EVENT_WINDOW_MS = 300;
  var DRIFT_TOLERANCE_SECONDS = 0.12;
  var groups = {};

  videos.forEach(function(video) {
    var groupName = video.dataset.syncGroup;
    if (!groupName) {
      return;
    }

    if (!groups[groupName]) {
      groups[groupName] = {
        videos: [],
        syncing: false,
        leader: null,
        rafId: null
      };
    }

    groups[groupName].videos.push(video);
  });

  function markProgrammatic(video) {
    video.dataset.syncProgrammaticUntil = String(Date.now() + PROGRAMMATIC_EVENT_WINDOW_MS);
  }

  function isProgrammatic(video) {
    var until = Number(video.dataset.syncProgrammaticUntil || 0);
    return Date.now() < until;
  }

  function withGroupSync(group, action) {
    if (group.syncing) {
      return;
    }

    group.syncing = true;
    try {
      action();
    } finally {
      group.syncing = false;
    }
  }

  function syncFollowerState(source, target, shouldPlay) {
    markProgrammatic(target);
    target.playbackRate = source.playbackRate;

    if (Math.abs(target.currentTime - source.currentTime) > DRIFT_TOLERANCE_SECONDS) {
      target.currentTime = source.currentTime;
    }

    if (shouldPlay) {
      target.play().catch(function() {
        // Ignore play rejections (for example, browser gesture policy).
      });
    }
  }

  function stepGroup(group) {
    if (!group.leader || group.leader.paused || group.leader.ended) {
      group.rafId = null;
      return;
    }

    withGroupSync(group, function() {
      group.videos.forEach(function(target) {
        if (target === group.leader) {
          return;
        }

        syncFollowerState(group.leader, target, !target.ended);
      });
    });

    group.rafId = window.requestAnimationFrame(function() {
      stepGroup(group);
    });
  }

  function ensurePlaybackLoop(group, leader) {
    group.leader = leader;
    if (group.rafId !== null) {
      return;
    }

    group.rafId = window.requestAnimationFrame(function() {
      stepGroup(group);
    });
  }

  function stopPlaybackLoop(group, leader) {
    if (group.leader !== leader) {
      return;
    }

    group.leader = null;
    if (group.rafId !== null) {
      window.cancelAnimationFrame(group.rafId);
      group.rafId = null;
    }
  }

  Object.keys(groups).forEach(function(groupName) {
    var group = groups[groupName];
    if (group.videos.length < 2) {
      return;
    }

    group.videos.forEach(function(video) {
      video.addEventListener('play', function() {
        if (isProgrammatic(video)) {
          return;
        }

        withGroupSync(group, function() {
          group.videos.forEach(function(target) {
            if (target === video) {
              return;
            }

            syncFollowerState(video, target, true);
          });
        });

        ensurePlaybackLoop(group, video);
      });

      video.addEventListener('pause', function() {
        if (isProgrammatic(video)) {
          return;
        }

        withGroupSync(group, function() {
          group.videos.forEach(function(target) {
            if (target === video) {
              return;
            }

            markProgrammatic(target);
            target.pause();
          });
        });

        stopPlaybackLoop(group, video);
      });

      video.addEventListener('seeking', function() {
        if (isProgrammatic(video)) {
          return;
        }

        withGroupSync(group, function() {
          group.videos.forEach(function(target) {
            if (target === video) {
              return;
            }

            markProgrammatic(target);
            target.currentTime = video.currentTime;
          });
        });
      });

      video.addEventListener('ratechange', function() {
        if (isProgrammatic(video)) {
          return;
        }

        withGroupSync(group, function() {
          group.videos.forEach(function(target) {
            if (target === video) {
              return;
            }

            markProgrammatic(target);
            target.playbackRate = video.playbackRate;
          });
        });
      });

      video.addEventListener('ended', function() {
        if (isProgrammatic(video)) {
          return;
        }

        stopPlaybackLoop(group, video);
      });
    });
  });
}

function setupSceneComparisonSelector() {
  var selector = document.getElementById('scene-selector');
  var sceneLabel = document.getElementById('scene-comparison-label');
  var asyncVideo = document.getElementById('asyncbev-scene-video');
  var asyncSource = document.getElementById('asyncbev-scene-source');
  var cmtVideo = document.getElementById('cmt-scene-video');
  var cmtSource = document.getElementById('cmt-scene-source');

  if (!selector || !sceneLabel || !asyncVideo || !asyncSource || !cmtVideo || !cmtSource) {
    return;
  }

  var sceneMap = {
    'scene-0003': {
      asyncbev: 'scene-0003',
      cmt: 'scene-003'
    },
    'scene-0016': {
      asyncbev: 'scene-0016',
      cmt: 'scene-0016'
    }
  };

  function applyScene(sceneName) {
    var sceneConfig = sceneMap[sceneName];
    if (!sceneConfig) {
      return;
    }

    var shouldResumePlayback = !asyncVideo.paused || !cmtVideo.paused;

    asyncVideo.pause();
    cmtVideo.pause();

    asyncVideo.dataset.syncGroup = sceneName;
    cmtVideo.dataset.syncGroup = sceneName;
    sceneLabel.textContent = sceneName;

    asyncSource.src = './static/video/asyncbev_async_10/' + sceneConfig.asyncbev + '/LIDAR_TOP_with_GT_vfr.mp4';
    cmtSource.src = './static/video/cmt_async_10/' + sceneConfig.cmt + '/LIDAR_TOP_with_GT_vfr.mp4';

    asyncVideo.load();
    cmtVideo.load();

    if (shouldResumePlayback) {
      Promise.all([
        asyncVideo.play().catch(function() {
          return null;
        }),
        cmtVideo.play().catch(function() {
          return null;
        })
      ]);
    }
  }

  selector.addEventListener('change', function() {
    applyScene(selector.value);
  });

  applyScene(selector.value);
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

    setupSceneComparisonSelector();
    setupSynchronizedVideos();

    bulmaSlider.attach();

})
