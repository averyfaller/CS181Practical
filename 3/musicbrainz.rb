require 'musicbrainz'
require 'csv'
require 'lastfm'
require 'httparty'
require 'pp'
require 'hashie'
require 'set'

MusicBrainz.configure do |c|
  # Application identity (required)
  c.app_name = "My Music App"
  c.app_version = "1.0"
  c.contact = "support@mymusicapp.com"

  # Cache config (optional)
  c.cache_path = "/tmp/musicbrainz-cache"
  c.perform_caching = true

  # Querying config (optional)
  c.query_interval = 1.2 # seconds
  c.tries_limit = 2
end

def get_top_tags(mbid)
  begin
    api_key = '7655e37476eb693cdf1a72ab8cc147cb'
    response = HTTParty.get("http://ws.audioscrobbler.com/2.0/?method=artist.gettoptags&mbid=#{mbid}&api_key=#{api_key}&format=json")
    Hashie::Mash.new(response.parsed_response).toptags.tag.select{ |x| x['count'] > 40 }.map(&:name).map(&:downcase)
  rescue NoMethodError => e
    puts response.parsed_response.inspect
    []
  end
end

def tag_string(tags, headers)
  output = Array.new(headers.length - 2, 0)
  tags.each do |tag|
    idx = headers.find_index(tag)
    output[idx - 2] = 1
  end
  output.join(',')
end

# pp get_top_tags('2e41ae9c-afd2-4f20-8f1e-17281ce9b472')

headers = Set.new(['artist', 'name'])
data = []

csv = CSV.read('data/artists.csv', :headers => true)
count = 0
csv.each do |row|
  count += 1
  # if count < 5
    # puts row.to_s.chomp.inspect
    tags = get_top_tags(row['artist'])
    data.append({
      'artist' => row['artist'],
      'name' => row['name'],
      'tags' => tags
    })
    # insert into headers
    headers.merge(tags)
    sleep(0.05)
  # end
end

File.open('data/output.csv', 'a') do |f|
  f.puts(headers.to_a.join(','))
  data.each do |row|
    f.puts("#{row['artist']},#{row['name']},#{tag_string(row['tags'], headers)}")
  end
end


