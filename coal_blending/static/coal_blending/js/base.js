$(document).ready(function(){
    /* Clock */
    setInterval(function(){
        $('#Clock').text(moment().format('ddd, D MMM Y, HH:mm:ss'));
    }, 1000)

    /* Enable Tooltip */
    $('[data-toggle="tooltip"]').tooltip();

});