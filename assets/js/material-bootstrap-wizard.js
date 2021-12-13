/*!

 =========================================================
 * Material Bootstrap Wizard - v1.0.2
 =========================================================

 * Product Page: https://www.creative-tim.com/product/material-bootstrap-wizard
 * Copyright 2017 Creative Tim (http://www.creative-tim.com)
 * Licensed under MIT (https://github.com/creativetimofficial/material-bootstrap-wizard/blob/master/LICENSE.md)

 =========================================================

 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 */

// Material Bootstrap Wizard Functions

var searchVisible = 0;
var transparent = true;
var mobile_device = false;

var service_url = "http://127.0.0.1:8000";

not_started_img="<img src=\"assets/img/not_started.png\"/>"
refresh_img="<img src=\"assets/img/refresh.png\"/>"

function run(){
  $("#result_img").empty();
  $("#result_img").prepend(refresh_img);
  $("#img_not_achievable").hide();
  $("#img_achievable").hide();
  $("#img_running").hide();

  if ($('input[type=radio][name=data_type]:checked').val() == "vision") {
    names = "vision_embeddings";
    dataset = $( "#tfds_vision" ).val();
  } else {
    names = "nlp_embeddings";
    dataset = $( "#tfds_nlp" ).val();
  }
  str_dataset = "dataset=" + dataset;
  console.log("Selected dataset: " + dataset)

  // get all the checked embeddings
  var embeddings = [];
  $.each($("input[name='" + names + "']:checked"), function(){
    embeddings.push($(this).val());
  });
  str_embeddings = "";
  if (embeddings.length > 0) {
    str_embeddings = "&embeddings=" + embeddings.join("&embeddings=");
  }

  label_noise = $("input[name=label_noise]").val();
  ln_appendix = "";
  if(label_noise != null && label_noise != "") {
    console.log("Querying with label noise: " + label_noise);
    ln_appendix = "&label_noise=" + label_noise;
  }

  $.ajax({
    url: service_url + "/check?" + str_dataset + str_embeddings + ln_appendix
    }).then(function(data) {
       console.log(data);
       if(data){
         console.log("Running or Done")
         target = $("input[name=accuracy]").val();
         console.log("Querying with target accuracy: " + target)
         $.ajax({
              url: service_url + "/get?target=" + target + "&" + str_dataset + str_embeddings + ln_appendix,
              success: function (data, textStatus, xhr) {
                state = data[0];
                img = data[2];
                if (state == "Pending") {
                  console.log("Pending");
                  //$(".wizard-footer").find('.btn-finish').show();
                  $("#result_img").empty();
                  $("#result_img").prepend(not_started_img);
                } else {
                  $("#result_img").empty();
                  $("#result_img").prepend(img);
                  if (state == "Achievable") {
                    $("#img_achievable").show();
                  } else if (state == "NotAchievable") {
                    $("#img_not_achievable").show();
                  } else {
                    $("#img_running").show();
                  }
                }
                console.log(data);
              },
              error: function (xhr, textStatus, errorThrown) {
                  console.log('Error in Operation');
              }
          });
       } else {
         console.log("Run")
         $.ajax({
              url: service_url + "/put?" + str_dataset + str_embeddings + ln_appendix,
              type: 'PUT',
              success: function (data, textStatus, xhr) {
                  console.log(data);
                  run()
              },
              error: function (xhr, textStatus, errorThrown) {
                  console.log('Error in Operation');
              }
          });
       }
    });
}

$(document).ready(function(){

  $.material.init();

  $('#datasets_vision').show();
  $('#datasets_nlp').hide();
  $('#vision_embeddings').show();
  $('#nlp_embeddings').hide();

  $('input[type=radio][name=data_type]').change(function() {
      if (this.value == 'vision') {
        $('#datasets_vision').show();
        $('#datasets_nlp').hide();
        $('#vision_embeddings').show();
        $('#nlp_embeddings').hide();
      }
      else {
        $('#datasets_vision').hide();
        $('#datasets_nlp').show();
        $('#vision_embeddings').hide();
        $('#nlp_embeddings').show();
      }
  });

  //set initial state.
  $('#textbox1').val(this.checked);

  $('#checkbox1').change(function() {
      if(this.checked) {
          var returnVal = confirm("Are you sure?");
          $(this).prop("checked", returnVal);
      }
      $('#textbox1').val(this.checked);
  });

  /*  Activate the tooltips      */
  $('[rel="tooltip"]').tooltip();

  // Code for the Validator
  var $validator = $('.wizard-card form').validate({
    rules: {
      accuracy: {
        required: true,
        number: true,
      },
      label_noise: {
        number: true,
      }
    },

    errorPlacement: function(error, element) {
      $(element).parent('div').addClass('has-error');
    }
  });

  // Wizard Initialization
  $('.wizard-card').bootstrapWizard({
    'tabClass': 'nav nav-pills',
    'nextSelector': '.btn-next',
    'previousSelector': '.btn-previous',

    onNext: function(tab, navigation, index) {
      var $valid = $('.wizard-card form').valid();
      if(!$valid) {
        $validator.focusInvalid();
        return false;
      }
    },

    onInit : function(tab, navigation, index){
      //check number of tabs and fill the entire row
      var $total = navigation.find('li').length;
      var $wizard = navigation.closest('.wizard-card');

      $first_li = navigation.find('li:first-child a').html();
      $moving_div = $('<div class="moving-tab">' + $first_li + '</div>');
      $('.wizard-card .wizard-navigation').append($moving_div);

      refreshAnimation($wizard, index);

      $('.moving-tab').css('transition','transform 0s');
    },

    onTabClick : function(tab, navigation, index){
      var $valid = $('.wizard-card form').valid();

      if(!$valid){
        return false;
      } else{
        return true;
      }
    },

    onTabShow: function(tab, navigation, index) {
      var $total = navigation.find('li').length;
      var $current = index+1;

      var $wizard = navigation.closest('.wizard-card');

      // If it's the last tab then hide the last button and show the finish instead
      if($current >= $total) {
        $($wizard).find('.btn-next').hide();
        $($wizard).find('.btn-finish').show();

        $($wizard).find('.btn-finish').off("click");
        $($wizard).find('.btn-finish').click(run);

        run();

      } else {
        $($wizard).find('.btn-next').show();
        $($wizard).find('.btn-finish').hide();
        $("#img_not_achievable").hide();
        $("#img_achievable").hide();
        $("#img_running").hide();
      }

      button_text = navigation.find('li:nth-child(' + $current + ') a').html();

      setTimeout(function(){
        $('.moving-tab').text(button_text);
      }, 150);

      var checkbox = $('.footer-checkbox');

      if( !index == 0 ){
        $(checkbox).css({
          'opacity':'0',
          'visibility':'hidden',
          'position':'absolute'
        });
      } else {
        $(checkbox).css({
          'opacity':'1',
          'visibility':'visible'
        });
      }

      refreshAnimation($wizard, index);
    }
  });


  // Prepare the preview for profile picture
  $("#wizard-picture").change(function(){
    readURL(this);
  });

  $('[data-toggle="wizard-radio"]').click(function(){
    wizard = $(this).closest('.wizard-card');
    wizard.find('[data-toggle="wizard-radio"]').removeClass('active');
    $(this).addClass('active');
    $(wizard).find('[type="radio"]').removeAttr('checked');
    $(this).find('[type="radio"]').attr('checked','true');
  });

  $('[data-toggle="wizard-checkbox"]').click(function(){
    if( $(this).hasClass('active')){
      $(this).removeClass('active');
      $(this).find('[type="checkbox"]').removeAttr('checked');
    } else {
      $(this).addClass('active');
      $(this).find('[type="checkbox"]').attr('checked','true');
    }
  });

  $('.set-full-height').css('height', 'auto');

});



//Function to show image before upload

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $('#wizardPicturePreview').attr('src', e.target.result).fadeIn('slow');
    }
    reader.readAsDataURL(input.files[0]);
  }
}

$(window).resize(function(){
  $('.wizard-card').each(function(){
    $wizard = $(this);

    index = $wizard.bootstrapWizard('currentIndex');
    refreshAnimation($wizard, index);

    $('.moving-tab').css({
      'transition': 'transform 0s'
    });
  });
});

function refreshAnimation($wizard, index){
  $total = $wizard.find('.nav li').length;
  $li_width = 100/$total;

  total_steps = $wizard.find('.nav li').length;
  move_distance = $wizard.width() / total_steps;
  index_temp = index;
  vertical_level = 0;

  mobile_device = $(document).width() < 600 && $total > 3;

  if(mobile_device){
    move_distance = $wizard.width() / 2;
    index_temp = index % 2;
    $li_width = 50;
  }

  $wizard.find('.nav li').css('width',$li_width + '%');

  step_width = move_distance;
  move_distance = move_distance * index_temp;

  $current = index + 1;

  if($current == 1 || (mobile_device == true && (index % 2 == 0) )){
    move_distance -= 8;
  } else if($current == total_steps || (mobile_device == true && (index % 2 == 1))){
    move_distance += 8;
  }

  if(mobile_device){
    vertical_level = parseInt(index / 2);
    vertical_level = vertical_level * 38;
  }

  $wizard.find('.moving-tab').css('width', step_width);
  $('.moving-tab').css({
    'transform':'translate3d(' + move_distance + 'px, ' + vertical_level +  'px, 0)',
    'transition': 'all 0.5s cubic-bezier(0.29, 1.42, 0.79, 1)'

  });
}

materialDesign = {

  checkScrollForTransparentNavbar: debounce(function() {
    if($(document).scrollTop() > 260 ) {
      if(transparent) {
        transparent = false;
        $('.navbar-color-on-scroll').removeClass('navbar-transparent');
      }
    } else {
      if( !transparent ) {
        transparent = true;
        $('.navbar-color-on-scroll').addClass('navbar-transparent');
      }
    }
  }, 17)

}

function debounce(func, wait, immediate) {
  var timeout;
  return function() {
    var context = this, args = arguments;
    clearTimeout(timeout);
    timeout = setTimeout(function() {
      timeout = null;
      if (!immediate) func.apply(context, args);
    }, wait);
    if (immediate && !timeout) func.apply(context, args);
  };
};
